import React, { useState } from 'react';
import './ControlPanel.css';

const ControlPanel = ({ 
  status, 
  onToggleTracking, 
  onResetTracking, 
  onToggleCrossCameraTracking, 
  onSetThreshold,
  onAddCamera 
}) => {
  const [threshold, setThreshold] = useState(status.similarity_threshold);
  const [cameraUrl, setCameraUrl] = useState('');
  const [addingCamera, setAddingCamera] = useState(false);
  const [addCameraError, setAddCameraError] = useState(null);
  
  const handleThresholdChange = (e) => {
    const value = parseFloat(e.target.value);
    setThreshold(value);
  };
  
  const handleThresholdApply = () => {
    onSetThreshold(threshold);
  };
  
  const handleCameraUrlChange = (e) => {
    setCameraUrl(e.target.value);
  };
  
  const handleAddCamera = async () => {
    if (!cameraUrl) {
      setAddCameraError('Please enter a camera URL');
      return;
    }
    
    setAddingCamera(true);
    setAddCameraError(null);
    
    try {
      const success = await onAddCamera(cameraUrl);
      if (success) {
        setCameraUrl('');
      } else {
        setAddCameraError('Failed to connect to camera');
      }
    } catch (err) {
      setAddCameraError('Failed to add camera: ' + err.message);
    } finally {
      setAddingCamera(false);
    }
  };
  
  return (
    <div className="control-panel">
      <h2>System Controls</h2>
      
      <div className="control-section">
        <h3>Tracking</h3>
        <div className="control-group">
          <button 
            onClick={onToggleTracking}
            className={status.tracking_enabled ? 'btn-active' : ''}
          >
            {status.tracking_enabled ? 'Disable Tracking' : 'Enable Tracking'}
          </button>
          <button onClick={onResetTracking}>Reset Tracking</button>
        </div>
      </div>
      
      <div className="control-section">
        <h3>Cross-Camera Tracking</h3>
        <div className="control-group">
          <button 
            onClick={onToggleCrossCameraTracking}
            className={status.cross_camera_tracking ? 'btn-active' : ''}
          >
            {status.cross_camera_tracking ? 'Disable' : 'Enable'}
          </button>
        </div>
      </div>
      
      <div className="control-section">
        <h3>Similarity Threshold: {threshold.toFixed(2)}</h3>
        <div className="control-group slider-group">
          <input 
            type="range" 
            min="0.5" 
            max="0.95" 
            step="0.01"
            value={threshold}
            onChange={handleThresholdChange}
          />
          <button onClick={handleThresholdApply}>Apply</button>
        </div>
      </div>
      
      <div className="control-section">
        <h3>Add IP Camera</h3>
        <div className="control-group">
          <input 
            type="text" 
            placeholder="rtsp://username:password@ip:port/stream"
            value={cameraUrl}
            onChange={handleCameraUrlChange}
          />
          <button 
            onClick={handleAddCamera}
            disabled={addingCamera}
          >
            {addingCamera ? 'Adding...' : 'Add Camera'}
          </button>
        </div>
        {addCameraError && <div className="error-message">{addCameraError}</div>}
      </div>
      
      <div className="system-info">
        <p>Active Cameras: {status.active_cameras.length}</p>
        <p>System Status: Online</p>
      </div>
    </div>
  );
};

export default ControlPanel; 