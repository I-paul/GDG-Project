import { db } from '../firebase';
import { collection, getDocs, query, where } from 'firebase/firestore';
import axios from 'axios'; // Assuming you're using axios for HTTP requests

const API_BASE_URL = 'http://localhost:5000'; 

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

// New function to send camera information to backend
export const sendCameraInfoToBackend = async (userId) => {
    try {
        // Fetch cameras for the user
        const cameras = await fetchCameras(userId);
        
        // Prepare payload to send to backend
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

        // Send information to backend
        const response = await axios.post(`${API_BASE_URL}/register_cameras`, payload);
        
        return response.data;
    } catch (error) {
        console.error('Error sending camera info to backend:', error);
        throw error;
    }
};