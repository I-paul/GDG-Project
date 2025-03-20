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

# Configuration - optimized for performance and low latency
DETECTION_CONFIDENCE = 0.35  # Lowered for better detection
MAX_FRAME_WIDTH = 400  # Further reduced for faster processing
DETECTION_INTERVAL = 0.03  # More frequent updates for better tracking
MIN_DETECTION_INTERVAL = 0.02  # Faster updates when person is tracked
FACE_DETECTION_INTERVAL = 0.08  # Faster face detection
MAX_FEATURE_AGE = 3.0  # Reduced to be more responsive to changes
ADAPTIVE_FEATURE_UPDATE = True  # Dynamically update features based on confidence

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

# Process frames in a separate thread to reduce latency
# Process frames in a separate thread to reduce latency
def process_frames(source_id, input_queue, output_queue):
    global selected_person_features, feature_cache, DETECTION_CONFIDENCE
    
    # Initialize tracker with optimized parameters for faster response
    tracker = DeepSort(max_age=10, n_init=1, nn_budget=100, embedder="mobilenet", 
                       embedder_gpu=True if device.type == 'cuda' else False)
    
    frame_count = 0
    last_detection_time = time.time()
    last_face_detection_time = time.time()
    detection_interval = DETECTION_INTERVAL
    
    # For adaptive processing
    person_detected = False
    consecutive_detections = 0
    
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
            
            # Dynamic detection interval - more frequent if tracking someone
            if tracking_enabled and selected_person_features is not None:
                if person_detected:
                    # Even faster updates when actively tracking
                    detection_interval = MIN_DETECTION_INTERVAL
                else:
                    detection_interval = DETECTION_INTERVAL
            else:
                detection_interval = DETECTION_INTERVAL * 1.5  # Slower updates when not tracking
            
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
                
                # Prioritize face detection when tracking is enabled
                if tracking_enabled and selected_person_features is not None:
                    # Run face detection more frequently when actively tracking
                    should_run_face_detection = current_time - last_face_detection_time >= FACE_DETECTION_INTERVAL
                else:
                    # Less frequent face detection when not tracking anyone
                    should_run_face_detection = current_time - last_face_detection_time >= FACE_DETECTION_INTERVAL * 2
                
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
                            scaleFactor=1.05,  # Lower for better detection
                            minNeighbors=4,    # Lower to catch more faces
                            minSize=(20, 20),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                    
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
                        # Adjust expansion based on image size for better proportions
                        expansion_factor = min(1.5, max(1.2, frame.shape[1] / 640))
                        y_expanded = max(0, y - int(h * 0.15))
                        h_expanded = int(h * expansion_factor)
                        
                        # Add face detection as a person detection
                        # Convert to XYXY format for compatibility with DeepSORT
                        # Higher confidence for face detections
                        detections.append(([x, y_expanded, x + w, y_expanded + h_expanded], 0.95, 0, None))
                    
                    last_face_detection_time = current_time
                
                # Always run person detection as a fallback, but optimize for selected tracking
                # Skip person detection every other frame if a face was detected
                should_run_person_detection = len(detections) == 0 or frame_count % 2 == 0
                
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
                    
                    track_data = {
                        'bbox': track.to_tlwh(),  # Convert to x,y,w,h format
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
                        
                        # Only mark as match if similarity exceeds threshold
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
                    
                    # Update the feature vector with the latest appearance using adaptive approach
                    if ADAPTIVE_FEATURE_UPDATE and selected_person_features is not None:
                        if best_similarity > 0.85:  # Very confident match
                            # Add to feature cache
                            feature_cache.append(best_match['features'])
                            if len(feature_cache) > MAX_FEATURE_CACHE:
                                feature_cache.pop(0)  # Remove oldest
                            
                            # Strong update - more weight to new features
                            selected_person_features = 0.6 * selected_person_features + 0.4 * best_match['features']
                            consecutive_detections += 1
                        elif best_similarity > 0.75:  # Good match
                            # Moderate update
                            selected_person_features = 0.8 * selected_person_features + 0.2 * best_match['features']
                            consecutive_detections += 1
                        else:  # Weak match
                            # Minimal update
                            selected_person_features = 0.95 * selected_person_features + 0.05 * best_match['features']
                        
                        # Normalize the feature vector
                        selected_person_features = selected_person_features / (norm(selected_person_features) + 1e-6)
                    
                    # Update detection interval based on tracking success
                    if not previous_person_detected and person_detected:
                        # Person just found - update more frequently
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
                    
                    # If cached features are very different, they might represent better appearance
                    if current_similarity < 0.9:
                        # Blend slightly with historical features
                        selected_person_features = 0.9 * selected_person_features + 0.1 * avg_features
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
    global selected_track_id, selected_person_features, tracking_enabled, global_preview_frame, feature_cache
    
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
mouse_callback_data = {source_id: {'source_id': source_id, 'frame': None, 'tracks': []} for source_id in caps.keys()}
for source_id in caps.keys():
    window_name = f"Camera {source_id}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_person, mouse_callback_data[source_id])

# Create control window with trackbars
control_window = "Camera Controls"
cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(control_window, 400, 300)

# Add trackbars for window size and matching threshold
cv2.createTrackbar("Window Width", control_window, args.window_width, 1280, lambda x: None)
cv2.createTrackbar("Window Height", control_window, args.window_height, 720, lambda x: None)
cv2.createTrackbar("Match Threshold %", control_window, int(args.similarity_threshold * 100), 100, lambda x: None)

# Create trackbars for additional controls
cv2.createTrackbar("Cross-Camera Tracking", control_window, 1, 1, lambda x: None)
cv2.createTrackbar("Show FPS", control_window, 1, 1, lambda x: None)
# Add detection sensitivity trackbar
cv2.createTrackbar("Detection Sensitivity", control_window, 35, 50, lambda x: None)

print("\n🔍 Controls:")
print("- Press 'q' to quit")
print("- Press 'r' to reset person tracking")
print("- Press 'a' to add a new IP camera (enter URL in console)")
print("- Press 't' to toggle tracking on/off")
print("- Press 'c' to toggle cross-camera tracking")
print("- Press 'f' to toggle FPS display")
print("- Use trackbars to adjust window size and matching threshold")
print("- Click on a person to track them\n")

last_fps_time = time.time()
frame_counts = {source_id: 0 for source_id in caps.keys()}
fps_values = {source_id: 0 for source_id in caps.keys()}

# Window layout management
def arrange_windows(source_ids, window_width, window_height):
    """Arrange windows in a grid layout"""
    num_windows = len(source_ids)
    if num_windows == 0:
        return
    
    screen_width = 1920  # Assumed screen width
    screen_height = 1080  # Assumed screen height
    
    # Calculate grid dimensions
    cols = min(3, num_windows)  # Max 3 columns
    rows = (num_windows + cols - 1) // cols
    
    # Calculate spacing and margins
    margin_x = 20
    margin_y = 40
    spacing = 10
    
    # Calculate available area
    available_width = screen_width - (margin_x * 2) - (spacing * (cols - 1))
    available_height = screen_height - (margin_y * 2) - (spacing * (rows - 1))
    
    # Calculate window size
    window_width = min(window_width, available_width // cols)
    window_height = min(window_height, available_height // rows)
    
    # Position windows
    for i, source_id in enumerate(source_ids):
        row = i // cols
        col = i % cols
        
        x = margin_x + col * (window_width + spacing)
        y = margin_y + row * (window_height + spacing)
        
        window_name = f"Camera {source_id}"
        cv2.moveWindow(window_name, x, y)
        cv2.resizeWindow(window_name, window_width, window_height)

# Create a priority queue for camera display
# Continuing from where the code was cut off

# Create a priority queue for camera display
def update_camera_priority(source_ids, latest_camera=None):
    """Reorder cameras to prioritize the one with the target"""
    # Put the latest camera first, if any
    ordered_sources = []
    if latest_camera in source_ids:
        ordered_sources.append(latest_camera)
    
    # Add the rest of the cameras
    for source_id in source_ids:
        if source_id != latest_camera:
            ordered_sources.append(source_id)
    
    return ordered_sources

# Main loop for capturing and processing frames
try:
    # Initialize variables for frame rate calculation
    start_time = time.time()
    frame_count = 0
    fps = 0
    last_fps_update = start_time
    
    # Keep track of which camera last detected the person
    latest_detection_camera = None
    
    while True:
        # Read trackbar values
        window_width = cv2.getTrackbarPos("Window Width", control_window)
        window_height = cv2.getTrackbarPos("Window Height", control_window)
        similarity_threshold = cv2.getTrackbarPos("Match Threshold %", control_window) / 100.0
        cross_camera_tracking = bool(cv2.getTrackbarPos("Cross-Camera Tracking", control_window))
        fps_display = bool(cv2.getTrackbarPos("Show FPS", control_window))
        detection_sensitivity = cv2.getTrackbarPos("Detection Sensitivity", control_window) / 100.0
        
        # Update detection confidence threshold
        DETECTION_CONFIDENCE = max(0.2, min(0.5, detection_sensitivity))
        
        # Update similarity threshold
        args.similarity_threshold = similarity_threshold
        
        # Arrange windows with updated size
        arrange_windows(caps.keys(), window_width, window_height)
        
        # Process frames from all cameras
        for source_id, cap in caps.items():
            ret, frame = cap.read()
            
            if not ret:
                # Try to reconnect if stream is lost
                print(f"⚠️ Lost connection to {source_id}, attempting to reconnect...")
                # For IP cameras, try reopening
                if source_id.startswith("ip_"):
                    caps[source_id].release()
                    time.sleep(0.5)
                    caps[source_id] = cv2.VideoCapture(all_sources[source_id])
                    ret, frame = caps[source_id].read()
                    if not ret:
                        # Skip this camera for this frame
                        continue
                else:
                    # Skip this camera for this frame
                    continue
            
            # Add frame to processing queue
            if not frame_queues[source_id].full():
                frame_queues[source_id].put(frame)
            
            # Get processed frame with detections
            try:
                if not result_queues[source_id].empty():
                    frame_with_tracks, tracks, best_match = result_queues[source_id].get(block=False)
                    
                    # Update mouse callback data
                    mouse_callback_data[source_id]['frame'] = frame_with_tracks
                    mouse_callback_data[source_id]['tracks'] = tracks
                    
                    # Calculate FPS for this camera
                    frame_counts[source_id] += 1
                    current_time = time.time()
                    if current_time - last_fps_time >= 1.0:
                        # Update FPS
                        fps_values[source_id] = frame_counts[source_id]
                        frame_counts[source_id] = 0
                        last_fps_time = current_time
                    
                    # Update which camera last saw the person
                    if best_match is not None:
                        latest_detection_camera = source_id
                    
                    # Draw bounding boxes and labels
                    for track in tracks:
                        x, y, w, h = map(int, track['bbox'])
                        track_id = track['track_id']
                        similarity = track.get('similarity', 0)
                        is_match = track.get('is_match', False)
                        
                        # Choose color based on match status
                        if is_match:
                            color = (0, 255, 0)  # Green for tracked person
                            thickness = 2
                        else:
                            color = (255, 0, 0)  # Blue for other people
                            thickness = 1
                        
                        # Draw bounding box
                        cv2.rectangle(frame_with_tracks, (x, y), (x + w, y + h), color, thickness)
                        
                        # Display track ID and similarity score if applicable
                        if is_match:
                            label = f"Person {track_id} ({similarity:.2f})"
                            cv2.putText(frame_with_tracks, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            label = f"ID: {track_id}"
                            cv2.putText(frame_with_tracks, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Display FPS if enabled
                    if fps_display:
                        cv2.putText(frame_with_tracks, f"FPS: {fps_values[source_id]}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Show tracking status
                    status_color = (0, 255, 0) if tracking_enabled else (0, 0, 255)
                    status_text = "Tracking: ON" if tracking_enabled else "Tracking: OFF"
                    cv2.putText(frame_with_tracks, status_text, 
                               (10, frame_with_tracks.shape[0] - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    # Show camera status
                    camera_text = f"Camera: {source_id}"
                    cv2.putText(frame_with_tracks, camera_text,
                               (10, frame_with_tracks.shape[0] - 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    # Show the frame
                    cv2.imshow(f"Camera {source_id}", frame_with_tracks)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error displaying camera {source_id}: {e}")
        
        # Reorder cameras to prioritize the one with the detected person
        if cross_camera_tracking and latest_detection_camera is not None:
            # Update camera order based on priority
            ordered_sources = update_camera_priority(caps.keys(), latest_detection_camera)
            
            # Apply the new order by rearranging windows
            arrange_windows(ordered_sources, window_width, window_height)
        
        # Show the preview of the selected person if available
        if global_preview_frame is not None:
            cv2.imshow("Tracked Person", global_preview_frame)
        
        # Update control window with current status
        control_frame = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Add tracking status info
        cv2.putText(control_frame, "Tracking Status:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        status_color = (0, 255, 0) if tracking_enabled else (0, 0, 255)
        status_text = "ENABLED" if tracking_enabled else "DISABLED"
        cv2.putText(control_frame, status_text, (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Add cross-camera status
        cv2.putText(control_frame, "Cross-Camera:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cross_color = (0, 255, 0) if cross_camera_tracking else (0, 0, 255)
        cross_text = "ENABLED" if cross_camera_tracking else "DISABLED"
        cv2.putText(control_frame, cross_text, (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cross_color, 2)
        
        # Add person ID info
        if selected_track_id is not None:
            cv2.putText(control_frame, f"Tracking ID: {selected_track_id}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(control_frame, "No person selected", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Add help text
        cv2.putText(control_frame, "Press 'q' to quit", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(control_frame, "Press 'r' to reset tracking", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(control_frame, "Press 't' to toggle tracking", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(control_frame, "Press 'a' to add IP camera", (10, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show control window
        cv2.imshow(control_window, control_frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 Exiting...")
            break
        elif key == ord('r'):
            # Reset tracking
            selected_track_id = None
            selected_person_features = None
            global_preview_frame = None
            feature_cache = []
            print("🔄 Reset tracking")
        elif key == ord('t'):
            # Toggle tracking
            tracking_enabled = not tracking_enabled
            status = "enabled" if tracking_enabled else "disabled"
            print(f"🔍 Tracking {status}")
        elif key == ord('c'):
            # Toggle cross-camera tracking
            cross_camera_tracking = not cross_camera_tracking
            cv2.setTrackbarPos("Cross-Camera Tracking", control_window, 1 if cross_camera_tracking else 0)
            status = "enabled" if cross_camera_tracking else "disabled"
            print(f"🔄 Cross-camera tracking {status}")
        elif key == ord('f'):
            # Toggle FPS display
            fps_display = not fps_display
            cv2.setTrackbarPos("Show FPS", control_window, 1 if fps_display else 0)
            status = "shown" if fps_display else "hidden"
            print(f"⏱️ FPS display {status}")
        elif key == ord('a'):
            # Add a new IP camera
            print("Enter the URL of the IP camera (e.g., rtsp://username:password@192.168.1.64:554/Streaming/Channels/1):")
            url = input().strip()
            if url:
                add_ip_camera(url)
        
        # Calculate overall FPS
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time

except KeyboardInterrupt:
    print("👋 Program terminated by user")
except Exception as e:
    print(f"❌ Error in main loop: {e}")
finally:
    # Clean up
    # Stop processing threads by sending None
    for source_id in processing_threads.keys():
        frame_queues[source_id].put(None)
    
    # Wait for threads to finish
    for source_id, thread in processing_threads.items():
        thread.join(timeout=1.0)
    
    # Release video captures
    for cap in caps.values():
        cap.release()
    
    # Close all windows
    cv2.destroyAllWindows()
    print("✅ Resources released. Exiting.")

if __name__ == "__main__":
    print("Multi-camera face tracking system terminated.")