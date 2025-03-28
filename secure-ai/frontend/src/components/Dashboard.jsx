import React, { useState, useEffect } from 'react';
import api from '../services/api';
import CameraGrid from './CameraGrid';
import ControlPanel from './ControlPanel';
import './Dashboard.css';

const Dashboard = () => {
  const [status, setStatus] = useState({
    tracking_enabled: true,
    cross_camera_tracking: true,
    similarity_threshold: 0.75,
    active_cameras: [],
    fps_display: true
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch initial status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        setLoading(true);
        const data = await api.getStatus();
        setStatus(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching status:', err);
        setError('Failed to connect to the tracking system. Please ensure the backend is running.');
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
    // Poll for updates every 2 seconds
    const intervalId = setInterval(fetchStatus, 2000);
    
    return () => clearInterval(intervalId);
  }, []);

  // Handle control actions
  const handleToggleTracking = async () => {
    try {
      const response = await api.toggleTracking();
      setStatus(prev => ({ ...prev, tracking_enabled: response.tracking_enabled }));
    } catch (err) {
      console.error('Error toggling tracking:', err);
    }
  };

  const handleResetTracking = async () => {
    try {
      await api.resetTracking();
      // Refresh status after reset
      const data = await api.getStatus();
      setStatus(data);
    } catch (err) {
      console.error('Error resetting tracking:', err);
    }
  };

  const handleToggleCrossCameraTracking = async () => {
    try {
      const response = await api.toggleCrossCameraTracking();
      setStatus(prev => ({ ...prev, cross_camera_tracking: response.cross_camera_tracking }));
    } catch (err) {
      console.error('Error toggling cross-camera tracking:', err);
    }
  };

  const handleSetThreshold = async (threshold) => {
    try {
      const response = await api.setThreshold(threshold);
      setStatus(prev => ({ ...prev, similarity_threshold: response.similarity_threshold }));
    } catch (err) {
      console.error('Error setting threshold:', err);
    }
  };

  const handleAddCamera = async (url) => {
    try {
      await api.addCamera(url);
      // Refresh camera list
      const data = await api.getStatus();
      setStatus(data);
    } catch (err) {
      console.error('Error adding camera:', err);
      return false;
    }
    return true;
  };

  if (loading) {
    return <div className="loading">Loading tracking system...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Multi-Camera Tracking System</h1>
      </header>
      
      <div className="dashboard-content">
        <ControlPanel 
          status={status}
          onToggleTracking={handleToggleTracking}
          onResetTracking={handleResetTracking}
          onToggleCrossCameraTracking={handleToggleCrossCameraTracking}
          onSetThreshold={handleSetThreshold}
          onAddCamera={handleAddCamera}
        />
        
        <CameraGrid 
          cameras={status.active_cameras} 
          trackingEnabled={status.tracking_enabled}
        />
      </div>
    </div>
  );
};

export default Dashboard; 