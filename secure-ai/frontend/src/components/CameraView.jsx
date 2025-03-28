import React, { useState, useEffect } from 'react';
import './CameraView.css';

const CameraView = ({ cameraId }) => {
  const [streamUrl, setStreamUrl] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Set stream URL based on camera ID
    setStreamUrl(`http://localhost:5000/api/camera/${cameraId}/stream`);
    
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [cameraId]);
  
  const handleImageError = () => {
    setError('Could not connect to camera stream');
  };
  
  // Format camera ID for display
  const formatCameraName = (id) => {
    if (id.startsWith('local_')) {
      return `Local Camera ${id.split('_')[1]}`;
    } else if (id.startsWith('ip_')) {
      return `IP Camera ${id.split('_')[1]}`;
    }
    return id;
  };
  
  return (
    <div className="camera-view">
      <div className="camera-header">
        <h3>{formatCameraName(cameraId)}</h3>
      </div>
      
      <div className="camera-feed">
        {isLoading ? (
          <div className="camera-loading">Loading camera feed...</div>
        ) : error ? (
          <div className="camera-error">{error}</div>
        ) : (
          <img 
            src={streamUrl} 
            alt={`Camera feed ${cameraId}`} 
            onError={handleImageError}
          />
        )}
      </div>
    </div>
  );
};

export default CameraView; 