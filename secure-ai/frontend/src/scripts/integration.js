import { db } from '../firebase';
import { collection, getDocs, query, where } from 'firebase/firestore';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000';

// Fetch user cameras from Firebase
export const fetchCameras = async (userId) => {
    try {
        const q = query(collection(db, 'Cameras'), where('owner', '==', userId));
        const querySnapshot = await getDocs(q);

        return querySnapshot.docs.map(doc => ({
            id: doc.id,
            ip: doc.data().ip,  // Assuming IP is stored in Firestore
            streamUrl: `${API_BASE_URL}/video_feed/${doc.id}`
        }));
    } catch (error) {
        console.error('Error fetching cameras:', error);
        return [];
    }
};

// Update camera viewer (avoid redundant calls)
export const updateCamViewer = async (setCameras, userId) => {
    try {
        const cameras = await fetchCameras(userId);
        setCameras(cameras);
        return cameras;
    } catch (error) {
        console.error('Error updating cam viewer:', error);
        setCameras([]);
        throw error;
    }
};

// Send camera data to backend with retry logic
export const sendCameraInfoToBackend = async (userId, cameras = null) => {
    try {
        // If cameras are not provided, fetch them (prevents redundant fetch)
        const cameraList = cameras || await fetchCameras(userId);

        if (cameraList.length === 0) {
            console.warn('No cameras found for user:', userId);
            return { message: 'No cameras to process' };
        }

        // Prepare payload
        const payload = {
            userId,
            cameras: cameraList.map(cam => ({
                id: cam.id,
                ip: cam.ip,
                streamUrl: cam.streamUrl
            }))
        };

        // Send data to backend
        const response = await axios.post(`${API_BASE_URL}/register_cameras`, payload, {
            timeout: 5000 // Set timeout for request
        });

        return response.data;
    } catch (error) {
        console.error('Error sending camera info to backend:', error);

        // Implement retry logic (optional)
        for (let i = 0; i < 2; i++) {
            try {
                console.log(`Retrying... attempt ${i + 1}`);
                const response = await axios.post(`${API_BASE_URL}/register_cameras`, payload);
                return response.data;
            } catch (retryError) {
                console.error(`Retry ${i + 1} failed:`, retryError);
            }
        }

        throw error;
    }
};

// ...existing code...

// Register cameras with backend
export const registerCameras = async (cameras) => {
    try {
        const response = await axios.post('http://localhost:5000/api/cameras', {
            cameras: cameras.map(cam => cam.ip)
        });
        return response.data.cameras;
    } catch (error) {
        console.error('Error registering cameras:', error);
        throw error;
    }
};

// Start tracking a person
export const startTracking = async (cameraId, trackId) => {
    try {
        const response = await axios.post('http://localhost:5000/api/track_person', {
            camera_id: cameraId,
            track_id: trackId
        });
        return response.data;
    } catch (error) {
        console.error('Error starting tracking:', error);
        throw error;
    }
};

// Stop tracking
export const stopTracking = async () => {
    try {
        const response = await axios.post('http://localhost:5000/api/stop_tracking');
        return response.data;
    } catch (error) {
        console.error('Error stopping tracking:', error);
        throw error;
    }
};

// Get camera status
export const getCameraStatus = async () => {
    try {
        const response = await axios.get('http://localhost:5000/api/camera_status');
        return response.data;
    } catch (error) {
        console.error('Error getting camera status:', error);
        throw error;
    }
};fc 
