import cv2
import torch
import numpy as np
import argparse
import threading
import queue
import time
from numpy.linalg import norm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort



# Parse command line arguments
parser = argparse.ArgumentParser(description='Multi-camera face tracking system with IP camera support')
parser.add_argument('--ip-cameras', nargs='+', type=str, default=[],
                    help='IP camera URLs (e.g., rtsp://username:password@192.168.1.64:554/Streaming/Channels/1)')
parser.add_argument('--buffer-size', type=int, default=1,  # Reduced buffer size for lower latency
                    help='Buffer size for frame preprocessing (smaller values reduce latency)')
parser.add_argument('--similarity-threshold', type=float, default=0.75,
                    help='Similarity threshold for face matching (0.0-1.0)')
parser.add_argument('--window-width', type=int, default=640,
                    help='Width of display window')
parser.add_argument('--window-height', type=int, default=480,
                    help='Height of display window')
parser.add_argument('--enable-gpu', action='store_true',
                    help='Enable GPU acceleration for YOLO and tracking')
args = parser.parse_args()

# Configure device based on availability and user preference
device = torch.device('cuda:0' if torch.cuda.is_available() and args.enable_gpu else 'cpu')
print(f"🖥️ Using device: {device}")

# Load YOLOv8 model - use a smaller model for faster processing
yolo_model = YOLO("yolov8n.pt")
if device.type == 'cuda':
    yolo_model.to(device)

# Frame processing queues and threads
frame_queues = {}
result_queues = {}
processing_threads = {}

# Store features of selected person
selected_person_features = None
selected_track_id = None
tracking_enabled = True  # Enable tracking by default
last_seen_times = {}  # Track when selected person was last seen on each camera
cross_camera_tracking = True  # Enable tracking across multiple cameras
fps_display = True  # Display FPS counter
time_last_updated = time.time()
global_preview_frame = None  # A single frame to preview the currently tracked person

# Configuration - optimized for stability and reduced flickering
DETECTION_CONFIDENCE = 0.35
MAX_FRAME_WIDTH = 400
# ANTI-FLICKER IMPROVEMENTS: More consistent intervals
DETECTION_INTERVAL = 0.1  # Increased detection interval for stability
MIN_DETECTION_INTERVAL = 0.08  # More consistent minimum detection
FACE_DETECTION_INTERVAL = 0.2  # Less frequent face detection to reduce flickering
MAX_FEATURE_AGE = 5.0  # Increased to maintain track stability
ADAPTIVE_FEATURE_UPDATE = True

# ANTI-FLICKER IMPROVEMENTS: Add bounding box smoothing
USE_BOX_SMOOTHING = True
smoothed_boxes = {}  # Dictionary to store smoothed boxes with track IDs as keys
SMOOTHING_FACTOR = 0.7  # Higher = smoother but less responsive

# Use a faster face detector for better performance
try:
    # Try MediaPipe for faster face detection
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    mp_face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    USE_MEDIAPIPE = True
    print("✅ Using MediaPipe for face detection (faster)")
except ImportError:
    USE_MEDIAPIPE = False
    print("⚠️ MediaPipe not available, falling back to OpenCV face detection")

# Load pretrained face detection model as fallback
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if face cascade loaded successfully and download if needed
if face_cascade.empty():
    print("⚠️ Face cascade not found. Downloading...")
    import urllib.request
    import os
    
    # Create directory if it doesn't exist
    cascade_dir = os.path.dirname(cv2.data.haarcascades)
    if not os.path.exists(cascade_dir):
        os.makedirs(cascade_dir)
    
    # Download the face cascade file
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    cascade_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
    urllib.request.urlretrieve(url, cascade_path)
    
    # Load again
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ Failed to load face cascade. Using regular person detection.")
        USE_FACE_DETECTION = True  # Still use YOLO detection
    else:
        print("✅ Face cascade successfully downloaded and loaded.")
        USE_FACE_DETECTION = True
else:
    print("✅ Face cascade loaded successfully.")
    USE_FACE_DETECTION = True

# Automatically detect available local cameras
def get_available_cameras(max_cams=10):
    available_cams = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    return available_cams

# Get all available local cameras
camera_sources = get_available_cameras()
print(f"✅ Detected Local Cameras: {camera_sources}")

# Add IP cameras from command line arguments
ip_cameras = args.ip_cameras
print(f"✅ Added IP Cameras: {len(ip_cameras)}")

# Create a mapping of all camera sources
all_sources = {}
# Add local cameras
for i, cam_id in enumerate(camera_sources):
    all_sources[f"local_{cam_id}"] = cam_id

# Add IP cameras
for i, ip_cam in enumerate(ip_cameras):
    all_sources[f"ip_{i}"] = ip_cam

if not all_sources:
    print("❌ No cameras detected! Exiting...")
    exit()

# Function to test if an IP camera stream is accessible with optimized settings
def test_ip_camera(url):
    try:
        # Try with RTSP-specific options for lower latency
        cap = cv2.VideoCapture(url)
        # Set buffer size to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Set lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            return ret
        return False
    except Exception as e:
        print(f"Error testing camera {url}: {e}")
        return False

# Function to detect faces using MediaPipe (faster than OpenCV)
def detect_faces_mediapipe(frame):
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detector.process(rgb_frame)
    faces = []
    
    if results.detections:
        h, w = frame.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            faces.append((x, y, width, height))
    
    return faces

# Feature cache for adaptive feature updates
feature_cache = []
MAX_FEATURE_CACHE = 10

# ANTI-FLICKER IMPROVEMENTS: Add box stabilization mechanism
def stabilize_box(track_id, new_box, alpha=0.3):
    """Smooth bounding box transitions to reduce flickering"""
    global smoothed_boxes
    
    if track_id not in smoothed_boxes:
        smoothed_boxes[track_id] = new_box
        return new_box
    
    # Get previous box
    prev_box = smoothed_boxes[track_id]
    
    # Apply exponential moving average
    x, y, w, h = prev_box
    new_x, new_y, new_w, new_h = new_box
    
    # Smooth out the coordinates and dimensions
    smooth_x = int(alpha * new_x + (1 - alpha) * x)
    smooth_y = int(alpha * new_y + (1 - alpha) * y)
    smooth_w = int(alpha * new_w + (1 - alpha) * w)
    smooth_h = int(alpha * new_h + (1 - alpha) * h)
    
    smoothed_box = (smooth_x, smooth_y, smooth_w, smooth_h)
    smoothed_boxes[track_id] = smoothed_box
    
    return smoothed_box

# ANTI-FLICKER IMPROVEMENTS: Add detection throttling mechanism
class DetectionThrottler:
    def __init__(self):
        self.last_face_detection = 0
        self.last_person_detection = 0
        self.detection_counts = {}
        self.last_method = None
    
    def should_run_face_detection(self, current_time, tracking=False):
        """Decide if face detection should run based on time and past results"""
        # Longer intervals when not tracking to save resources
        interval = FACE_DETECTION_INTERVAL * (0.8 if tracking else 2.0)
        
        # Consistently alternate between detection methods
        if self.last_method == "face" and current_time - self.last_face_detection < interval * 2:
            return False
            
        if current_time - self.last_face_detection >= interval:
            self.last_face_detection = current_time
            self.last_method = "face"
            return True
        return False
    
    def should_run_person_detection(self, current_time, frame_count, faces_found=0):
        # If faces were found, we can skip person detection sometimes
        if faces_found > 0 and frame_count % 3 != 0:
            return False
            
        # Consistently alternate between detection methods
        if self.last_method == "person" and current_time - self.last_person_detection < DETECTION_INTERVAL:
            return False
            
        if current_time - self.last_person_detection >= DETECTION_INTERVAL:
            self.last_person_detection = current_time
            self.last_method = "person"
            return True
        return False

# Process frames in a separate thread to reduce latency
def process_frames(source_id, input_queue, output_queue):
    global selected_person_features, feature_cache, DETECTION_CONFIDENCE
    
    # Initialize tracker with optimized parameters for faster response
    tracker = DeepSort(max_age=15, n_init=2, nn_budget=100, embedder="mobilenet", 
                      embedder_gpu=True if device.type == 'cuda' else False)
    
    frame_count = 0
    last_detection_time = time.time()
    last_face_detection_time = time.time()
    detection_interval = DETECTION_INTERVAL
    
    # For adaptive processing
    person_detected = False
    consecutive_detections = 0
    
    # Initialize detection throttler to reduce flickering
    throttler = DetectionThrottler()
    
    # Track the last known detection to fill in gaps (anti-flicker)
    last_detection_result = None
    detection_gap_counter = 0
    MAX_DETECTION_GAP = 5  # Maximum number of frames to keep showing previous detection
    
    while True:
        try:
            # Get frame from queue with timeout
            try:
                frame = input_queue.get(timeout=0.1)  # Shorter timeout
            except queue.Empty:
                continue
                
            if frame is None:  # Signal to exit
                break
                
            frame_count += 1
            current_time = time.time()
            
            # Dynamic detection interval - more consistent for stability
            if tracking_enabled and selected_person_features is not None:
                detection_interval = MIN_DETECTION_INTERVAL
            else:
                detection_interval = DETECTION_INTERVAL
            
            # Only run detection on detection intervals
            if current_time - last_detection_time >= detection_interval:
                # Resize frame for faster processing
                orig_h, orig_w = frame.shape[:2]
                scale_factor = min(1.0, MAX_FRAME_WIDTH / orig_w)
                if scale_factor < 1.0:
                    width = int(orig_w * scale_factor)
                    height = int(orig_h * scale_factor)
                    frame_resized = cv2.resize(frame, (width, height))
                else:
                    frame_resized = frame
                    
                detections = []
                faces_found = 0
                
                # ANTI-FLICKER: More consistent face detection timing
                should_run_face_detection = throttler.should_run_face_detection(
                    current_time, 
                    tracking=(tracking_enabled and selected_person_features is not None)
                )
                
                # Run face detection if enabled and time
                if USE_FACE_DETECTION and should_run_face_detection:
                    if USE_MEDIAPIPE:
                        # Use MediaPipe face detection (faster)
                        faces = detect_faces_mediapipe(frame_resized)
                    else:
                        # Use OpenCV's built-in face detector with optimized parameters
                        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,  # Increased for more stability
                            minNeighbors=5,   # Increased to reduce false positives
                            minSize=(25, 25), # Increased minimum size
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                    
                    faces_found = len(faces)
                    for face in faces:
                        if isinstance(face, tuple):
                            x, y, w, h = face
                        else:
                            x, y, w, h = face
                        
                        # Scale back to original frame size if resized
                        if scale_factor < 1.0:
                            x = int(x / scale_factor)
                            y = int(y / scale_factor)
                            w = int(w / scale_factor)
                            h = int(h / scale_factor)
                        
                        # Expand face box to include upper body for better tracking
                        # ANTI-FLICKER: More consistent body box expansion
                        expansion_factor = 1.3  # Fixed expansion factor for consistency
                        y_expanded = max(0, y - int(h * 0.1))
                        h_expanded = int(h * expansion_factor)
                        
                        # Add face detection as a person detection
                        # Convert to XYXY format for compatibility with DeepSORT
                        # Higher confidence for face detections
                        detections.append(([x, y_expanded, x + w, y_expanded + h_expanded], 0.95, 0, None))
                
                # ANTI-FLICKER: More consistent person detection timing
                should_run_person_detection = throttler.should_run_person_detection(
                    current_time, frame_count, faces_found
                )
                
                # Always run person detection as a fallback, but with more consistent timing
                if should_run_person_detection:
                    results = yolo_model(frame_resized)[0]
                    
                    # Fix: Properly handle YOLO detection results
                    for box in results.boxes:
                        # Convert tensor to numpy for proper indexing
                        xyxy = box.xyxy.cpu().numpy() if hasattr(box.xyxy, 'cpu') else box.xyxy
                        conf = box.conf.cpu().numpy() if hasattr(box.conf, 'cpu') else box.conf
                        cls = box.cls.cpu().numpy() if hasattr(box.cls, 'cpu') else box.cls
                        
                        if len(xyxy) > 0:  # Ensure box has coordinates
                            x1, y1, x2, y2 = map(int, xyxy[0])
                            confidence = float(conf[0])
                            class_id = int(cls[0])
                            
                            # Only detect people (Class ID = 0)
                            if class_id == 0 and confidence > DETECTION_CONFIDENCE:
                                # Scale back to original frame size if resized
                                if scale_factor < 1.0:
                                    x1 = int(x1 / scale_factor)
                                    y1 = int(y1 / scale_factor)
                                    x2 = int(x2 / scale_factor)
                                    y2 = int(y2 / scale_factor)
                                
                                # Add person detection in XYXY format
                                detections.append(([x1, y1, x2, y2], confidence, class_id, None))
                
                # ANTI-FLICKER: If no detections but we had recent ones, use the last result
                if len(detections) == 0 and last_detection_result is not None and detection_gap_counter < MAX_DETECTION_GAP:
                    # Use previous detection to avoid flickering
                    detections = last_detection_result
                    detection_gap_counter += 1
                else:
                    # We have new detections, update the last known result
                    if len(detections) > 0:
                        last_detection_result = detections
                        detection_gap_counter = 0
                
                # Update the tracker with bounding boxes in the correct format
                tracked_objects = tracker.update_tracks(detections, frame=frame)
                last_detection_time = current_time
                
                # Process results and check for face matches
                result_with_tracks = []
                best_match = None
                best_similarity = 0.0
                
                # Reset person detection flag
                previous_person_detected = person_detected
                person_detected = False
                
                for track in tracked_objects:
                    if not track.is_confirmed():
                        continue
                    
                    # Get box in TLWH format
                    bbox = track.to_tlwh()
                    
                    # ANTI-FLICKER: Apply box smoothing for confirmed tracks
                    if USE_BOX_SMOOTHING:
                        # Convert to tuple for smoothing
                        box_tuple = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        smoothed_bbox = stabilize_box(track.track_id, box_tuple, alpha=SMOOTHING_FACTOR)
                        bbox = np.array(smoothed_bbox)
                    
                    track_data = {
                        'bbox': bbox,  # Now contains smoothed bbox
                        'track_id': track.track_id,
                        'features': track.features[-1] if track.features is not None and len(track.features) > 0 else None,
                        'is_match': False,
                        'similarity': 0.0
                    }
                    
                    # If tracking is enabled and we have a selected person, check for matches
                    if selected_person_features is not None and tracking_enabled and track_data['features'] is not None:
                        # Compute cosine similarity with selected person features
                        similarity = np.dot(selected_person_features, track_data['features']) / (
                            norm(selected_person_features) * norm(track_data['features']) + 1e-6)
                        
                        track_data['similarity'] = similarity
                        
                        # ANTI-FLICKER: Use more consistent threshold application
                        if similarity > args.similarity_threshold:
                            track_data['is_match'] = True
                            person_detected = True
                            
                            # Keep track of the best match
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = track_data
                    
                    result_with_tracks.append(track_data)
                
                # Store the time this person was last seen on this camera
                if best_match is not None:
                    last_seen_times[source_id] = current_time
                    
                    # ANTI-FLICKER: More gradual feature updates
                    if ADAPTIVE_FEATURE_UPDATE and selected_person_features is not None:
                        # Add to feature cache regardless of similarity
                        feature_cache.append(best_match['features'])
                        if len(feature_cache) > MAX_FEATURE_CACHE:
                            feature_cache.pop(0)  # Remove oldest
                        
                        # Update with gradually increasing weight based on confidence
                        if best_similarity > 0.9:  # Very confident match
                            # Still use conservative update (was 0.6/0.4)
                            selected_person_features = 0.85 * selected_person_features + 0.15 * best_match['features']
                            consecutive_detections += 1
                        elif best_similarity > 0.8:  # Good match
                            # More conservative update (was 0.8/0.2)
                            selected_person_features = 0.9 * selected_person_features + 0.1 * best_match['features']
                            consecutive_detections += 1
                        else:  # Weak match
                            # Minimal update (was 0.95/0.05)
                            selected_person_features = 0.98 * selected_person_features + 0.02 * best_match['features']
                        
                        # Normalize the feature vector
                        selected_person_features = selected_person_features / (norm(selected_person_features) + 1e-6)
                    
                    # Update detection interval based on tracking success
                    if not previous_person_detected and person_detected:
                        # Person just found - update more frequently but not too fast
                        detection_interval = MIN_DETECTION_INTERVAL
                    elif consecutive_detections > 5:
                        # Stable tracking - can slightly reduce update frequency
                        detection_interval = MIN_DETECTION_INTERVAL * 1.2
                
                # If lost tracking but have cached features, use them as fallback
                elif selected_person_features is not None and tracking_enabled and len(feature_cache) > 0 and person_detected == False:
                    # Calculate average of recent features
                    avg_features = np.mean(feature_cache, axis=0)
                    avg_features = avg_features / (norm(avg_features) + 1e-6)
                    
                    # Calculate similarity of current features with cached
                    current_similarity = np.dot(selected_person_features, avg_features) / (
                        norm(selected_person_features) * norm(avg_features) + 1e-6)
                    
                    # ANTI-FLICKER: More conservative feature update
                    if current_similarity < 0.9:
                        # Even more conservative blend (was 0.9/0.1)
                        selected_person_features = 0.95 * selected_person_features + 0.05 * avg_features
                        selected_person_features = selected_person_features / (norm(selected_person_features) + 1e-6)
                    
                    consecutive_detections = 0
                
                output_queue.put((frame.copy(), result_with_tracks, best_match))
            else:
                # If skipping detection, still return frame without tracks
                output_queue.put((frame.copy(), [], None))
                
            input_queue.task_done()
        except Exception as e:
            print(f"Error in processing thread for {source_id}: {e}")
            continue

# Open video streams and create processing threads
caps = {}
active_streams = 0

for source_id, source in all_sources.items():
    # Create queues for frame processing
    frame_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
    result_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
    
    # For local cameras, use the camera ID directly
    if source_id.startswith("local_"):
        caps[source_id] = cv2.VideoCapture(source)
        
        if caps[source_id].isOpened():
            # Optimized camera settings for lower latency
            caps[source_id].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            caps[source_id].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps[source_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer
            caps[source_id].set(cv2.CAP_PROP_FPS, 30)  # Higher FPS
            
            # Set camera format to MJPG for faster processing
            caps[source_id].set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Start processing thread
            processing_threads[source_id] = threading.Thread(
                target=process_frames,
                args=(source_id, frame_queues[source_id], result_queues[source_id]),
                daemon=True
            )
            processing_threads[source_id].start()
            active_streams += 1
        else:
            print(f"❌ Failed to open local camera {source}")
            del caps[source_id]
            
    # For IP cameras, use the URL
    elif source_id.startswith("ip_"):
        if test_ip_camera(source):
            caps[source_id] = cv2.VideoCapture(source)
            
            # Optimize RTSP connection for real-time processing
            if source.startswith('rtsp'):
                caps[source_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Use TCP for more reliable streaming
                caps[source_id].set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TCP)
                # Set lower resolution for faster processing
                caps[source_id].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                caps[source_id].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            # Start processing thread
            processing_threads[source_id] = threading.Thread(
                target=process_frames,
                args=(source_id, frame_queues[source_id], result_queues[source_id]),
                daemon=True
            )
            processing_threads[source_id].start()
            active_streams += 1
            print(f"✅ Successfully connected to IP camera: {source}")
        else:
            print(f"❌ Failed to connect to IP camera: {source}")

if not caps:
    print("❌ No cameras could be opened! Exiting...")
    exit()

print(f"🎥 Successfully opened {active_streams} camera streams")

# Mouse click event to select a person
def select_person(event, x, y, flags, param):
    global selected_track_id, selected_person_features, tracking_enabled, global_preview_frame, feature_cache, smoothed_boxes
    
    source_id = param['source_id']
    frame = param.get('frame')
    tracks = param.get('tracks', [])
    
    if event == cv2.EVENT_LBUTTONDOWN and frame is not None and tracks:
        # Find if clicked on any tracked person
        for track in tracks:
            x1, y1, w, h = map(int, track['bbox'])
            if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                selected_track_id = track['track_id']
                
                # Store features for tracking
                if track.get('features') is not None:
                    # Reset feature cache when selecting new person
                    feature_cache = []
                    selected_person_features = track.get('features')
                    tracking_enabled = True
                    
                    # ANTI-FLICKER: Clear previous smoothed boxes
                    smoothed_boxes = {}
                    
                    print(f"🎯 Selected Person ID: {selected_track_id} from {source_id}")
                    
                    # Create a preview of the selected person
                    x1, y1, w, h = map(int, track['bbox'])
                    if 0 <= x1 < frame.shape[1] and 0 <= y1 < frame.shape[0] and \
                       x1 + w <= frame.shape[1] and y1 + h <= frame.shape[0]:
                        person_crop = frame[y1:y1+h, x1:x1+w].copy()
                        # Resize to a standard size
                        person_crop = cv2.resize(person_crop, (100, 150))
                        global_preview_frame = person_crop
                    
                    # Reset last seen times
                    last_seen_times.clear()
                    last_seen_times[source_id] = time.time()
                    break
                else:
                    print(f"⚠️ No features available for this person. Try selecting again.")

# Function to add a new IP camera during runtime
def add_ip_camera(url):
    global active_streams
    
    if test_ip_camera(url):
        source_id = f"ip_{len([k for k in all_sources.keys() if k.startswith('ip_')])}"
        all_sources[source_id] = url
        caps[source_id] = cv2.VideoCapture(url)
        
        # Optimize RTSP settings
        if url.startswith('rtsp'):
            caps[source_id].set(cv2.CAP_PROP_BUFFERSIZE, 1)
            caps[source_id].set(cv2.CAP_PROP_RTSP_TRANSPORT, cv2.CAP_RTSP_TCP)
            caps[source_id].set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            caps[source_id].set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create frame processing queues
        frame_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
        result_queues[source_id] = queue.Queue(maxsize=args.buffer_size)
        
        # Start processing thread
        processing_threads[source_id] = threading.Thread(
            target=process_frames,
            args=(source_id, frame_queues[source_id], result_queues[source_id]),
            daemon=True
        )
        processing_threads[source_id].start()
        active_streams += 1
        
        # Create window and set mouse callback
        window_name = f"Camera {source_id}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, select_person, {'source_id': source_id, 'frame': None, 'tracks': []})
        
        print(f"✅ Added new IP camera: {url} as {source_id}")
        return True
    else:
        print(f"❌ Failed to connect to IP camera: {url}")
        return False

# Set mouse callback for each camera window with shared frame data
# Set mouse callback for each camera window with shared frame data
mouse_callback_data = {}
for source_id in caps.keys():
    window_name = f"Camera {source_id}"
    cv2.namedWindow(window_name)
    mouse_callback_data[source_id] = {'source_id': source_id, 'frame': None, 'tracks': []}
    cv2.setMouseCallback(window_name, select_person, mouse_callback_data[source_id])

# Create a window for tracked person preview if tracking enabled
preview_window_name = "Tracked Person"
cv2.namedWindow(preview_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(preview_window_name, 150, 200)

# FPS calculation variables
fps_start_time = time.time()
fps_frame_count = 0
fps_values = {}

# Main loop - capture frames and display results
try:
    while True:
        # Process all cameras
        for source_id, cap in caps.items():
            try:
                # Read frame from camera with error handling
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Failed to read frame from {source_id}, attempting to reconnect...")
                    if source_id.startswith("ip_"):
                        # Attempt to reconnect to IP camera
                        url = all_sources[source_id]
                        cap.release()
                        caps[source_id] = cv2.VideoCapture(url)
                        if caps[source_id].isOpened():
                            print(f"✅ Successfully reconnected to {source_id}")
                        continue
                    elif source_id.startswith("local_"):
                        # Attempt to reconnect to local camera
                        cam_id = all_sources[source_id]
                        cap.release()
                        caps[source_id] = cv2.VideoCapture(cam_id)
                        if caps[source_id].isOpened():
                            print(f"✅ Successfully reconnected to {source_id}")
                        continue
                
                # Put frame in processing queue
                if not frame_queues[source_id].full():
                    frame_queues[source_id].put(frame)
                
                # Try to get processed results without blocking
                try:
                    processed_frame, tracks, best_match = result_queues[source_id].get_nowait()
                    result_queues[source_id].task_done()
                    
                    # Update mouse callback data
                    mouse_callback_data[source_id]['frame'] = processed_frame
                    mouse_callback_data[source_id]['tracks'] = tracks
                    
                    # Calculate FPS for this camera
                    if source_id not in fps_values:
                        fps_values[source_id] = {'start_time': time.time(), 'frames': 0, 'fps': 0}
                    
                    fps_values[source_id]['frames'] += 1
                    elapsed = time.time() - fps_values[source_id]['start_time']
                    if elapsed >= 1.0:  # Update FPS every second
                        fps_values[source_id]['fps'] = fps_values[source_id]['frames'] / elapsed
                        fps_values[source_id]['frames'] = 0
                        fps_values[source_id]['start_time'] = time.time()
                    
                    # Draw tracking results
                    window_name = f"Camera {source_id}"
                    display_frame = processed_frame.copy()
                    
                    # Add title showing camera source
                    title = f"{source_id}"
                    if source_id.startswith("ip_"):
                        title = f"IP Camera {source_id.split('_')[1]}"
                    elif source_id.startswith("local_"):
                        title = f"Local Camera {source_id.split('_')[1]}"
                    
                    # Draw title bar
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 30), (45, 45, 45), -1)
                    cv2.putText(display_frame, title, (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Display FPS if enabled
                    if fps_display:
                        fps_text = f"FPS: {fps_values[source_id]['fps']:.1f}"
                        cv2.putText(display_frame, fps_text, (display_frame.shape[1] - 120, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Draw tracking boxes with target indicators
                    for track in tracks:
                        x, y, w, h = map(int, track['bbox'])
                        track_id = track['track_id']
                        
                        # Choose color based on matching or random for each track
                        if track.get('is_match', False) and tracking_enabled and selected_person_features is not None:
                            # Matched person - green with intensity based on confidence
                            similarity = track.get('similarity', 0)
                            # Brighter green for higher confidence matches
                            green_intensity = min(255, int(128 + 127 * similarity))
                            color = (0, green_intensity, 0)
                            thickness = 2
                            
                            # Add similarity percentage to label
                            label = f"ID: {track_id} ({similarity:.2f})"
                        else:
                            # Different color for each track ID (but consistent)
                            color_seed = abs(hash(str(track_id))) % 256
                            color = ((color_seed * 1777) % 256, (color_seed * 2777) % 256, (color_seed * 3777) % 256)
                            thickness = 1
                            label = f"ID: {track_id}"
                        
                        # Draw bounding box and label
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
                        
                        # Add label with background for better visibility
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_width = label_size[0] + 10
                        cv2.rectangle(display_frame, (x, y - 20), (x + label_width, y), color, -1)
                        cv2.putText(display_frame, label, (x + 5, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # For tracked person, draw additional indicators
                        if track.get('is_match', False) and tracking_enabled and selected_person_features is not None:
                            # Draw target corners for better visibility
                            corner_length = min(20, min(w, h) // 3)
                            
                            # Top-left corner
                            cv2.line(display_frame, (x, y), (x + corner_length, y), (0, 255, 255), 2)
                            cv2.line(display_frame, (x, y), (x, y + corner_length), (0, 255, 255), 2)
                            
                            # Top-right corner
                            cv2.line(display_frame, (x + w, y), (x + w - corner_length, y), (0, 255, 255), 2)
                            cv2.line(display_frame, (x + w, y), (x + w, y + corner_length), (0, 255, 255), 2)
                            
                            # Bottom-left corner
                            cv2.line(display_frame, (x, y + h), (x + corner_length, y + h), (0, 255, 255), 2)
                            cv2.line(display_frame, (x, y + h), (x, y + h - corner_length), (0, 255, 255), 2)
                            
                            # Bottom-right corner
                            cv2.line(display_frame, (x + w, y + h), (x + w - corner_length, y + h), (0, 255, 255), 2)
                            cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 255), 2)
                            
                            # Update preview window if this is the best match
                            if best_match is not None and best_match['track_id'] == track_id:
                                person_crop = processed_frame[y:y+h, x:x+w].copy()
                                if person_crop.size > 0:  # Ensure we have a valid crop
                                    global_preview_frame = cv2.resize(person_crop, (100, 150))
                    
                    # Add tracking status indicator
                    status_text = "TRACKING ON" if tracking_enabled else "TRACKING OFF"
                    status_color = (0, 255, 0) if tracking_enabled else (0, 0, 255)
                    cv2.putText(display_frame, status_text, (10, display_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                    
                    # Add last seen indicator if tracking is enabled
                    if tracking_enabled and selected_person_features is not None:
                        if source_id in last_seen_times:
                            time_since_seen = time.time() - last_seen_times[source_id]
                            if time_since_seen < 5.0:  # Recently seen
                                seen_text = "PERSON PRESENT"
                                seen_color = (0, 255, 0)
                            else:
                                seen_text = f"Last seen: {time_since_seen:.1f}s ago"
                                seen_color = (0, 165, 255)  # Orange
                        else:
                            seen_text = "Not seen yet"
                            seen_color = (0, 0, 255)  # Red
                        
                        cv2.putText(display_frame, seen_text, (display_frame.shape[1] - 200, display_frame.shape[0] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, seen_color, 2)
                    
                    # Show the frame
                    cv2.imshow(window_name, display_frame)
                
                except queue.Empty:
                    # No results yet, skip this camera for now
                    pass
            
            except Exception as e:
                print(f"Error processing camera {source_id}: {e}")
                # Try to recover by releasing and re-opening the camera
                try:
                    if source_id in caps:
                        caps[source_id].release()
                        if source_id.startswith("ip_"):
                            url = all_sources[source_id]
                            caps[source_id] = cv2.VideoCapture(url)
                        elif source_id.startswith("local_"):
                            cam_id = all_sources[source_id]
                            caps[source_id] = cv2.VideoCapture(cam_id)
                except Exception as e2:
                    print(f"Failed to recover camera {source_id}: {e2}")
        
        # Display preview of tracked person if available
        if global_preview_frame is not None and tracking_enabled:
            # Create an info panel with tracking stats
            preview_display = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Add the person image
            h, w = global_preview_frame.shape[:2]
            preview_display[10:10+h, 50:50+w] = global_preview_frame
            
            # Add tracking info
            if selected_track_id is not None:
                cv2.putText(preview_display, f"ID: {selected_track_id}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Count cameras where the person is visible
                visible_count = sum(1 for t in last_seen_times.values() if time.time() - t < 5.0)
                cv2.putText(preview_display, f"Visible on: {visible_count}/{len(caps)} cams", 
                           (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow(preview_window_name, preview_display)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Toggle tracking with 't' key
        if key == ord('t'):
            tracking_enabled = not tracking_enabled
            print(f"🔍 Tracking {'Enabled' if tracking_enabled else 'Disabled'}")
        
        # Reset tracking with 'r' key
        elif key == ord('r'):
            selected_person_features = None
            selected_track_id = None
            tracking_enabled = True
            global_preview_frame = None
            feature_cache = []
            smoothed_boxes = {}
            last_seen_times.clear()
            print("🔄 Tracking Reset")
        
        # Toggle cross-camera tracking with 'c' key
        elif key == ord('c'):
            cross_camera_tracking = not cross_camera_tracking
            print(f"🔀 Cross-Camera Tracking {'Enabled' if cross_camera_tracking else 'Disabled'}")
        
        # Toggle FPS display with 'f' key
        elif key == ord('f'):
            fps_display = not fps_display
            print(f"⏱️ FPS Display {'Enabled' if fps_display else 'Disabled'}")
        
        # Add new IP camera with 'a' key
        elif key == ord('a'):
            print("📹 Enter RTSP URL for new camera (e.g., rtsp://user:pass@192.168.1.100:554/stream):")
            cv2.destroyWindow("Input")
            cv2.namedWindow("Input")
            cv2.moveWindow("Input", 100, 100)
            
            # Create a simple input dialog
            input_frame = np.zeros((100, 600, 3), dtype=np.uint8)
            cv2.putText(input_frame, "Type URL and press Enter. ESC to cancel.", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("Input", input_frame)
            
            # Collect keyboard input
            ip_url = ""
            collecting_input = True
            while collecting_input:
                input_frame = np.zeros((100, 600, 3), dtype=np.uint8)
                cv2.putText(input_frame, "Type URL and press Enter. ESC to cancel.", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                cv2.putText(input_frame, ip_url, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Input", input_frame)
                
                k = cv2.waitKey(0) & 0xFF
                if k == 27:  # ESC key
                    collecting_input = False
                elif k == 13:  # Enter key
                    if ip_url:
                        add_ip_camera(ip_url)
                    collecting_input = False
                elif k == 8:  # Backspace
                    ip_url = ip_url[:-1]
                elif 32 <= k <= 126:  # Printable ASCII
                    ip_url += chr(k)
            
            cv2.destroyWindow("Input")
        
        # Adjust similarity threshold with up/down arrows
        elif key == 82:  # Up arrow
            args.similarity_threshold = min(0.95, args.similarity_threshold + 0.05)
            print(f"⬆️ Similarity Threshold: {args.similarity_threshold:.2f}")
        elif key == 84:  # Down arrow
            args.similarity_threshold = max(0.5, args.similarity_threshold - 0.05)
            print(f"⬇️ Similarity Threshold: {args.similarity_threshold:.2f}")
        
        # Save tracked person image with 's' key
        elif key == ord('s') and global_preview_frame is not None:
            timestamp = int(time.time())
            filename = f"tracked_person_{timestamp}.jpg"
            cv2.imwrite(filename, global_preview_frame)
            print(f"💾 Saved tracked person image to {filename}")
        
        # Exit with 'q' or ESC key
        elif key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    print("👋 Program interrupted by user")

finally:
    # Clean up resources
    print("🧹 Cleaning up...")
    
    # Signal threads to exit and join
    for source_id in frame_queues:
        try:
            frame_queues[source_id].put(None)  # Signal to exit
            if source_id in processing_threads:
                processing_threads[source_id].join(timeout=1.0)
        except Exception as e:
            print(f"Error closing thread for {source_id}: {e}")
    
    # Release all camera captures
    for source_id, cap in caps.items():
        try:
            cap.release()
        except Exception as e:
            print(f"Error releasing camera {source_id}: {e}")
    
    # Close all windows
    cv2.destroyAllWindows()
    print("✅ Program terminated successfully")