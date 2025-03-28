import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const api = {
  // Get system status
  getStatus: async () => {
    const response = await axios.get(`${API_BASE_URL}/status`);
    return response.data;
  },
  
  // Get list of cameras
  getCameras: async () => {
    const response = await axios.get(`${API_BASE_URL}/cameras`);
    return response.data.cameras;
  },
  
  // Toggle tracking
  toggleTracking: async () => {
    const response = await axios.post(`${API_BASE_URL}/toggle_tracking`);
    return response.data;
  },
  
  // Reset tracking
  resetTracking: async () => {
    const response = await axios.post(`${API_BASE_URL}/reset_tracking`);
    return response.data;
  },
  
  // Toggle cross-camera tracking
  toggleCrossCameraTracking: async () => {
    const response = await axios.post(`${API_BASE_URL}/toggle_cross_camera`);
    return response.data;
  },
  
  // Set similarity threshold
  setThreshold: async (threshold) => {
    const response = await axios.post(`${API_BASE_URL}/set_threshold`, { threshold });
    return response.data;
  },
  
  // Add a new camera
  addCamera: async (url) => {
    const response = await axios.post(`${API_BASE_URL}/add_camera`, { url });
    return response.data;
  }
};

export default api; 