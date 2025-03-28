import React, { useState, useEffect } from 'react';
import './TrackedPersonView.css';

const TrackedPersonView = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);
  
  const handleImageError = () => {
    setError('No person currently being tracked');
  };
  
  return (
    <div className="tracked-person-view">
      {isLoading ? (
        <div className="preview-loading">Loading preview...</div>
      ) : error ? (
        <div className="preview-error">{error}</div>
      ) : (
        <img 
          src="http://localhost:5000/api/preview" 
          alt="Tracked person preview" 
          onError={handleImageError}
        />
      )}
    </div>
  );
};

export default TrackedPersonView; 