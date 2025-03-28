import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import { updateCamViewer,sendCameraInfoToBackend } from '../scripts/integration';
import { 
  ArrowLeft, 
  Camera, 
  ChevronDown, 
  ChevronUp, 
  RefreshCw, 
  AlertTriangle 
} from 'lucide-react';
import './styling/camview.css';

const CamViewer = () => {
    const navigate = useNavigate();
    const [cameras, setCameras] = useState([]);
    const [expandedCamera, setExpandedCamera] = useState(null);
    const [errorMessages, setErrorMessages] = useState({});
    const [loading, setLoading] = useState(true);
    const [cameraLayout, setCameraLayout] = useState('auto');
    const webcamRefs = useRef({});
    const auth = getAuth();

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (user) => {
            if (user) {
                setLoading(true);
                try {
                    // Update camera viewer
                    await updateCamViewer(setCameras, user.uid);
                    
                    // Send camera info to backend
                    await sendCameraInfoToBackend(user.uid);
                    
                    setLoading(false);
                } catch (error) {
                    console.error('Error in camera setup:', error);
                    setLoading(false);
                }
            } else {
                navigate('/login');
            }
        });
        return () => unsubscribe();
    }, [auth, navigate]);

    useEffect(() => {
        let socket;
        let retryAttempts = 0;
        const maxRetries = 5;

        const connectWebSocket = () => {
            socket = new WebSocket('ws://localhost:8765');

            socket.onopen = () => {
                console.log('WebSocket connection established.');
                retryAttempts = 0; // Reset retry attempts on successful connection
            };

            socket.onmessage = (event) => {
                try {
                    const framesData = JSON.parse(event.data);
                    setCameras((prevCameras) =>
                        prevCameras.map((camera) => ({
                            ...camera,
                            analyzedFrame: framesData[camera.id] || camera.analyzedFrame,
                        }))
                    );
                } catch (err) {
                    console.error('Error parsing WebSocket message:', err);
                }
            };

            socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            socket.onclose = (event) => {
                console.log(`WebSocket connection closed (Code: ${event.code}, Reason: ${event.reason}).`);
                if (event.code === 1011) {
                    console.error('Server encountered an unexpected error. Stopping retries.');
                    return; // Stop retrying for server-side errors
                }
                if (retryAttempts < maxRetries) {
                    retryAttempts++;
                    console.log(`Retrying WebSocket connection (${retryAttempts}/${maxRetries})...`);
                    setTimeout(connectWebSocket, 2000); // Retry after 2 seconds
                } else {
                    console.error('Max WebSocket retry attempts reached. Connection failed.');
                }
            };
        };

        connectWebSocket();

        return () => {
            if (socket) {
                socket.close();
            }
        };
    }, []);

    // Determine camera layout based on number of cameras
    useEffect(() => {
        if (cameras.length === 1) {
            setCameraLayout('single-camera');
        } else if (cameras.length >= 2 && cameras.length <= 4) {
            setCameraLayout('multi-camera');
        } else if (cameras.length > 4) {
            setCameraLayout('many-cameras');
        } else {
            setCameraLayout('auto');
        }
    }, [cameras]);

    // Function to expand a selected camera view
    const expandCamera = (cameraId) => {
        setExpandedCamera(expandedCamera === cameraId ? null : cameraId);
    };

    // Function to retry loading a failed camera feed
    const handleRetry = (cameraId) => {
        setErrorMessages((prevErrors) => ({ ...prevErrors, [cameraId]: null }));
        setLoading(true);
        updateCamViewer(setCameras, auth.currentUser?.uid)
            .then(() => setLoading(false))
            .catch(() => setLoading(false));
    };

    // Render loading state
    if (loading) {
        return (
            <section className='viewer'>
                <div className="viewer-container">
                    <div className="camera-feed-loading"></div>
                </div>
            </section>
        );
    }

    return (
        <section className='viewer'>
            <div className="viewer-container">
                <button className="back-btn" onClick={() => navigate('/')}> 
                    <ArrowLeft size={18} /> Back to Management
                </button>
                <div className="viewer-header">
                    <h2>Live AI-Processed <span className="highlight">Camera Feeds</span></h2>
                    <p>View and track activity across all connected cameras in real-time.</p>
                </div>
                
                <div className={`camera-grid ${cameraLayout}`}>
                    {cameras.length === 0 ? (
                        <div className="no-cameras-message">
                            <AlertTriangle size={48} color="#f44336" />
                            <p>No cameras connected. Add a camera to get started.</p>
                        </div>
                    ) : (
                        cameras.map((camera) => (
                            <div 
                                key={camera.id} 
                                className={`camera-item ${camera.status.toLowerCase()}`}
                            >
                                <div 
                                    className="camera-header" 
                                    onClick={() => expandCamera(camera.id)}
                                >
                                    <div className="camera-title">
                                        <h3>{camera.name}</h3>
                                        <span 
                                            className={`status-indicator ${camera.status.toLowerCase()}`}
                                        >
                                            {camera.status}
                                        </span>
                                    </div>
                                    {expandedCamera === camera.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                </div>
                                
                                {errorMessages[camera.id] ? (
                                    <div className="camera-error">
                                        <AlertTriangle size={32} color="#f44336" />
                                        <p>Error loading feed</p>
                                        <button 
                                            onClick={() => handleRetry(camera.id)}
                                            className="retry-btn"
                                        >
                                            <RefreshCw size={16} /> Retry
                                        </button>
                                    </div>
                                ) : camera.status.toLowerCase() === 'online' ? (
                                    <div className="camera-feed">
                                        {camera.analyzedFrame ? (
                                            <img
                                                src={`data:image/jpeg;base64,${camera.analyzedFrame}`}
                                                alt={`Analyzed feed for ${camera.name}`}
                                                style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                            />
                                        ) : (
                                            <p>Loading analyzed feed...</p>
                                        )}
                                        <div className="live-indicator">LIVE</div>
                                    </div>
                                ) : (
                                    <div className="camera-offline">
                                        <Camera size={48} color="#adb5bd" />
                                        <p>Camera Offline</p>
                                    </div>
                                )}

                                {expandedCamera === camera.id && (
                                    <div className="dropdown-panel">
                                        <div className="detailed-info">
                                            <div className="info-row">
                                                <span className="info-label">Location</span>
                                                <span className="info-value">{camera.location || 'Not specified'}</span>
                                            </div>
                                            <div className="info-row">
                                                <span className="info-label">Last Online</span>
                                                <span className="info-value">
                                                    {camera.lastOnline || 'Unknown'}
                                                </span>
                                            </div>
                                        </div>
                                        <div className="camera-controls">
                                            <button 
                                                className="control-btn expand-btn"
                                                title="Expand Camera"
                                            >
                                                <ChevronUp size={16} />
                                            </button>
                                            <button 
                                                className="control-btn track-btn"
                                                title="Track Activity"
                                            >
                                                <Camera size={16} />
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))
                    )}
                </div>

                {cameras.length > 0 && (
                    <div className="viewer-footer">
                        <p>
                            Connected Cameras: <span className="connected-count">
                                {cameras.filter(c => c.status.toLowerCase() === 'online').length} / {cameras.length}
                            </span>
                        </p>
                        <button className="add-camera-btn">
                            <Camera size={16} /> Add New Camera
                        </button>
                    </div>
                )}
            </div>
        </section>
    );
};

export default CamViewer;