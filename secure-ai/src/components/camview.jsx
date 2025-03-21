import React, { useRef, useState, useEffect } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import { ArrowLeft, Camera, UserCheck, Settings } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './styling/camview.css';

const CamViewer = () => {
    const viewerRef = useRef();
    const navigate = useNavigate();
    const [cameras, setCameras] = useState([]);
    const [expandedCamera, setExpandedCamera] = useState(null);
    
    gsap.registerPlugin(useGSAP, ScrollTrigger);
    
    // Load cameras from localStorage on component mount
    useEffect(() => {
        const storedCameras = localStorage.getItem('cameras');
        if (storedCameras) {
            setCameras(JSON.parse(storedCameras));
        } else {
            // Fallback to default cameras if none in localStorage
            setCameras([
                { id: 1, name: 'Front Door Camera', status: 'Online', location: 'Front Door' },
                { id: 2, name: 'Back Yard Camera', status: 'Offline', location: 'Back Yard' },
                { id: 3, name: 'Living Room Camera', status: 'Online', location: 'Living Room' },
                { id: 4, name: 'Garage Camera', status: 'Online', location: 'Garage' }
            ]);
        }
    }, []);
    
    // Function to handle camera expansion
    const expandCamera = (cameraId) => {
        if (expandedCamera === cameraId) {
            setExpandedCamera(null); // Collapse if already expanded
        } else {
            setExpandedCamera(cameraId); // Expand clicked camera
        }
    };
    
    // Navigate back to add camera page
    const handleBackToManagement = () => {
        navigate('/');
    };
    
    // Navigate to add camera form
    const handleAddCamera = () => {
        navigate('/');
    };
    
    return (
        <section className='viewer' ref={viewerRef}>
            <div className="viewer-container">
                <div className="viewer-header">
                    <button className="back-btn" onClick={handleBackToManagement}>
                        <ArrowLeft size={18} />
                        Back to Management
                    </button>
                    <h2>Live Camera <span className="highlight">Monitoring</span></h2>
                    <p>View and track activity across all your connected cameras in real-time.</p>
                </div>
                
                <div className="camera-grid">
                    {cameras.map((camera) => (
                        <div 
                            key={camera.id} 
                            className={`camera-item ${camera.status.toLowerCase()}`}
                            onClick={() => camera.status.toLowerCase() === 'online' && expandCamera(camera.id)}
                        >
                            <div className="camera-feed">
                                {/* Placeholder for actual camera feed */}
                                <div className="camera-placeholder">
                                    {camera.status.toLowerCase() === 'online' ? (
                                        <>
                                            <div className="placeholder-icon"></div>
                                            <div className="live-indicator">LIVE</div>
                                        </>
                                    ) : (
                                        <div className="offline-message">Camera Offline</div>
                                    )}
                                </div>
                                
                                {/* This is where you would embed the actual stream */}
                                {/* <iframe src="camera-stream-url" title={camera.name} className="camera-stream"></iframe> */}
                            </div>
                            
                            <div className="camera-info">
                                <h3>{camera.name}</h3>
                                <p className="camera-location">{camera.location}</p>
                                <div className={`status-indicator ${camera.status.toLowerCase()}`}>
                                    <span className="status-dot"></span>
                                    <span className="status-text">{camera.status}</span>
                                </div>
                            </div>
                            
                            {camera.status.toLowerCase() === 'online' && (
                                <div className="camera-controls">
                                    <button className="control-btn expand-btn">
                                        <span className="icon">⤢</span>
                                    </button>
                                    <button className="control-btn track-btn">
                                        <UserCheck size={16} />
                                    </button>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
                
                <div className="viewer-footer">
                    <p>Connected Cameras: <span className="connected-count">
                        {cameras.filter(c => c.status.toLowerCase() === 'online').length}/{cameras.length}
                    </span></p>
                    <button className="add-camera-btn" onClick={handleAddCamera}>
                        <span className="icon">+</span> Add Camera
                    </button>
                </div>
            </div>
            
            {/* Expanded camera view */}
            {expandedCamera && (
                <div className="expanded-overlay">
                    <div className="expanded-camera">
                        <div className="expanded-header">
                            <h3>{cameras.find(c => c.id === expandedCamera)?.name}</h3>
                            <button className="close-btn" onClick={() => setExpandedCamera(null)}>×</button>
                        </div>
                        <div className="expanded-feed">
                            {/* Placeholder for actual expanded camera feed */}
                            <div className="expanded-placeholder">
                                <div className="placeholder-icon large"></div>
                                <div className="live-indicator">LIVE</div>
                            </div>
                            
                            {/* This is where you would embed the actual expanded stream */}
                            {/* <iframe src="camera-stream-url" title={cameras.find(c => c.id === expandedCamera)?.name} className="expanded-stream"></iframe> */}
                        </div>
                        <div className="expanded-controls">
                            <button className="expanded-control-btn">
                                <UserCheck size={18} /> Track Person
                            </button>
                            <button className="expanded-control-btn">
                                <Camera2 size={18} /> Screenshot
                            </button>
                            <button className="expanded-control-btn">
                                <Settings size={18} /> Settings
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
};

export default CamViewer;