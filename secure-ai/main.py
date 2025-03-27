import cv2
import torch
import numpy as np
import argparse
import threading
import queue
import time
import base64
import json

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# WebSocket server for streaming
import asyncio
import websockets

from flask import Flask, request, jsonify

# Parse command line arguments
parser = argparse.ArgumentParser(description='AI-powered multi-camera video processing')
parser.add_argument('--ip-cameras', nargs='+', type=str, required=True,
                    help='IP camera URLs (e.g., rtsp://username:password@192.168.1.64:554/Streaming/Channels/1)')
parser.add_argument('--buffer-size', type=int, default=5,
                    help='Buffer size for frame preprocessing')
parser.add_argument('--detection-confidence', type=float, default=0.35,
                    help='Minimum confidence for object detection')
parser.add_argument('--websocket-port', type=int, default=8765,
                    help='WebSocket server port')
args = parser.parse_args()

# Configure device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv8 model for object detection
yolo_model = YOLO("yolov8n.pt")
if device.type == 'cuda':
    yolo_model.to(device)

# Frame processing class
class CameraProcessor:
    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.frame_queue = queue.Queue(maxsize=args.buffer_size)
        self.result_queue = queue.Queue(maxsize=args.buffer_size)
        self.tracker = DeepSort(max_age=15)
        self.is_running = False
        self.thread = None

    def start(self):
        """Start camera stream and processing thread"""
        self.is_running = True
        self.thread = threading.Thread(target=self._process_stream, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop camera stream and processing thread"""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _process_stream(self):
        """Continuous frame capture and processing"""
        cap = cv2.VideoCapture(self.camera_url)
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame from {self.camera_url}")
                break

            # Skip queue if full to prevent blocking
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            # Process frames
            try:
                processed_frame = self._process_frame(frame)
                
                # Skip result queue if full
                try:
                    if not self.result_queue.full():
                        self.result_queue.put_nowait(processed_frame)
                except queue.Full:
                    pass
            except Exception as e:
                print(f"Error processing frame: {e}")

        cap.release()

    def _process_frame(self, frame):
        """Apply AI processing to a single frame"""
        # Resize for faster processing
        orig_h, orig_w = frame.shape[:2]
        scale_factor = min(1.0, 640 / orig_w)
        
        if scale_factor < 1.0:
            width = int(orig_w * scale_factor)
            height = int(orig_h * scale_factor)
            frame_resized = cv2.resize(frame, (width, height))
        else:
            frame_resized = frame

        # Detect objects using YOLO
        results = yolo_model(frame_resized)[0]
        detections = []

        for box in results.boxes:
            xyxy = box.xyxy.cpu().numpy()[0]
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])

            # Only consider people
            if cls == 0 and conf > args.detection_confidence:
                # Scale back to original frame size
                if scale_factor < 1.0:
                    x1, y1, x2, y2 = [int(coord / scale_factor) for coord in xyxy]
                else:
                    x1, y1, x2, y2 = map(int, xyxy)

                detections.append(([x1, y1, x2, y2], conf, cls, None))

        # Track objects
        tracked_objects = self.tracker.update_tracks(detections, frame=frame)

        # Draw tracking results
        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            bbox = track.to_tlwh()
            x, y, w, h = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

# WebSocket server to stream processed frames
async def websocket_server(processors, port):
    async def handler(websocket, path):
        try:
            while True:
                # Collect processed frames from all cameras
                frames_data = {}
                for camera_id, processor in processors.items():
                    try:
                        # Get latest processed frame without blocking
                        frame = processor.result_queue.get_nowait()
                        
                        # Encode frame as base64
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        frames_data[camera_id] = frame_base64
                    except queue.Empty:
                        # No frame available for this camera
                        pass
                
                # Send frames if any are available
                if frames_data:
                    await websocket.send(json.dumps(frames_data))
                
                # Short sleep to control transmission rate
                await asyncio.sleep(0.1)
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")

    server = await websockets.serve(handler, "0.0.0.0", port)
    print(f"WebSocket server started on port {port}")
    await server.wait_closed()

app = Flask(__name__)

# Store camera information
camera_data = {}

@app.route('/register_cameras', methods=['POST'])
def register_cameras():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        user_id = data.get('userId')
        camera_details = data.get('cameraDetails', [])

        # Store camera details in memory
        camera_data[user_id] = camera_details

        # Log the received data
        print(f"Received camera data for user {user_id}: {camera_details}")

        # Return success response
        return jsonify({"message": "Cameras registered successfully"}), 200
    except Exception as e:
        print(f"Error in /register_cameras: {e}")
        return jsonify({"error": "Failed to register cameras"}), 500

# Run Flask server in a separate thread
def run_flask_server():
    app.run(host='0.0.0.0', port=5000)

# Main execution
def main():
    # Wait for camera data from the frontend
    while not camera_data:
        print("Waiting for camera data from frontend...")
        time.sleep(2)

    # Create processors for each camera
    processors = {}
    for user_id, cameras in camera_data.items():
        for camera in cameras:
            camera_url = camera['streamUrl']
            processor = CameraProcessor(camera_url)
            processor.start()
            processors[camera['cameraId']] = processor

    # Start WebSocket server
    try:
        asyncio.run(websocket_server(processors, args.websocket_port))
    except KeyboardInterrupt:
        print("Stopping server...")
    finally:
        # Stop all camera processors
        for processor in processors.values():
            processor.stop()

if __name__ == "__main__":
    main()