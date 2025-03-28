import { db } from '../firebase';
import { collection, getDocs, query, where } from 'firebase/firestore';
import axios from 'axios'; // Assuming you're using axios for HTTP requests

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000'; // Use Vite's environment variable system

// Fetch cameras from Firebase
export const fetchCameras = async (userId) => {
    try {
        const userCamerasRef = collection(db, 'Cameras');
        const q = query(userCamerasRef, where('owner', '==', userId));
        const querySnapshot = await getDocs(q);
        
        return querySnapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data(),
            aiStreamUrl: `${API_BASE_URL}/video_feed/${doc.id}` 
        }));
    } catch (error) {
        console.error('Error fetching cameras:', error);
        return [];  
    }
};

// Update camera viewer with fetched cameras
export const updateCamViewer = async (setCameras, userId) => {
    try {
        const cameras = await fetchCameras(userId);
        setCameras(cameras.map(cam => ({
            ...cam,
            streamUrl: cam.aiStreamUrl 
        })));
        return cameras;
    } catch (error) {
        console.error('Error updating cam viewer:', error);
        setCameras([]);
        throw error;
    }
};

// Send camera information to backend
export const sendCameraInfoToBackend = async (userId) => {
    try {
        const cameras = await fetchCameras(userId);
        const payload = {
            userId: userId,
            totalCameras: cameras.length,
            cameraDetails: cameras.map(camera => ({
                cameraId: camera.id,
                name: camera.name,
                location: camera.location,
                streamUrl: camera.aiStreamUrl
            }))
        };

        const response = await axios.post(`${API_BASE_URL}/register_cameras`, payload);
        return response.data;
    } catch (error) {
        if (error.code === 'ERR_NETWORK') {
            console.error('Network error: Unable to reach the backend. Please check the API_BASE_URL or server status.');
        } else {
            console.error('Error sending camera info to backend:', error);
        }
        throw error;
    }
};

// Send camera data to WebSocket server
export const sendCameraDataToWebSocket = (cameraDetails) => {
    const socket = new WebSocket('ws://localhost:8765');
    socket.onopen = () => {
        console.log('WebSocket connection established.');
        // Ensure the data is sent in the correct format
        const payload = {
            cameraId: cameraDetails.cameraId,
            name: cameraDetails.name,
            location: cameraDetails.location,
            streamUrl: cameraDetails.streamUrl,
        };
        socket.send(JSON.stringify(payload));
    };
    socket.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
    socket.onclose = (event) => {
        console.log(`WebSocket connection closed (Code: ${event.code}, Reason: ${event.reason}).`);
    };
};