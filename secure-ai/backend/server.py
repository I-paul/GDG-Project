from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/register_cameras', methods=['POST'])
def register_cameras():
    data = request.json
    print(f"Received camera registration data: {data}")
    return jsonify({"status": "success", "message": "Cameras registered successfully!"})

@app.route('/video_feed/<camera_id>', methods=['GET'])
def video_feed(camera_id):
    # Example endpoint for video feed
    return f"Streaming video for camera {camera_id}"

if __name__ == '__main__':
    try:
        print("Starting backend server. Press Ctrl+C to stop.")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
