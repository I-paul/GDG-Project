from flask import Flask, request, jsonify, Response
from flask_cors import CORS  
import cv2  # For video feed handling
from main import analyze_frame  # Import AI model function

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store registered cameras
registered_cameras = {}

@app.route('/register_cameras', methods=['POST'])
def register_cameras():
    data = request.json
    camera_id = data.get("camera_id")
    camera_url = data.get("camera_url")
    
    if not camera_id or not camera_url:
        return jsonify({"status": "error", "message": "Camera ID and URL are required!"}), 400
    
    registered_cameras[camera_id] = camera_url
    print(f"Registered camera: {camera_id} with URL: {camera_url}")
    return jsonify({"status": "success", "message": "Cameras registered successfully!"})

@app.route('/video_feed/<camera_id>', methods=['GET'])
def video_feed(camera_id):
    if camera_id not in registered_cameras:
        return jsonify({"status": "error", "message": "Camera not found!"}), 404
    
    camera_url = registered_cameras[camera_id]
    cap = cv2.VideoCapture(camera_url)
    
    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Analyze the frame using the AI model
            analyzed_frame = analyze_frame(frame)
            
            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', analyzed_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("Starting backend server. Press Ctrl+C to stop.")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
