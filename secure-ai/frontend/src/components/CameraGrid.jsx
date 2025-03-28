import React from 'react';
import CameraView from './CameraView';
import TrackedPersonView from './TrackedPersonView';
import './CameraGrid.css';

const CameraGrid = ({ cameras, trackingEnabled }) => {
  return (
    <div className="camera-grid">
      {cameras.length === 0 ? (
        <div className="no-cameras">
          <p>No cameras connected. Add a camera using the control panel.</p>
        </div>
      ) : (
        <>
          {cameras.map(cameraId => (
            <CameraView 
              key={cameraId} 
              cameraId={cameraId} 
            />
          ))}
        </>
      )}
      
      {trackingEnabled && (
        <div className="preview-panel">
          <h3>Tracked Person</h3>
          <TrackedPersonView />
        </div>
      )}
    </div>
  );
};

export default CameraGrid; 