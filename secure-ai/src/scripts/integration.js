import { db } from '../firebase';
import { collection, getDocs, query, where } from 'firebase/firestore';

const API_BASE_URL = 'http://your-backend-ip:5000'; 

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