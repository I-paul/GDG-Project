import React, { useRef, useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import { updateCamViewer } from '../scripts/integration';
import { ArrowLeft, Camera, ChevronDown, ChevronUp } from 'lucide-react';
import './styling/camview.css';

const CamViewer = () => {
    const navigate = useNavigate();
    const [cameras, setCameras] = useState([]);
    const [expandedCamera, setExpandedCamera] = useState(null);
    const [errorMessages, setErrorMessages] = useState({});
    const [loading, setLoading] = useState(true);
    const webcamRefs = useRef({});
    const auth = getAuth();

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            if (user) {
                updateCamViewer(setCameras, user.uid);
            } else {
                navigate('/login');
            }
        });
        return () => unsubscribe();
    }, [auth, navigate]);

    // Function to expand a selected camera view
    const expandCamera = (cameraId) => {
        setExpandedCamera(expandedCamera === cameraId ? null : cameraId);
    };

    // Function to retry loading a failed camera feed
    const handleRetry = (cameraId) => {
        setErrorMessages((prevErrors) => ({ ...prevErrors, [cameraId]: null }));
        updateCamViewer(setCameras, auth.currentUser?.uid);
    };

    return (
        <section className='viewer'>
            <div className="viewer-container">
                <button className="back-btn" onClick={() => navigate('/')}> 
                    <ArrowLeft size={18} /> Back to Management
                </button>
                <h2>Live AI-Processed Camera Feeds</h2>
                <p>View and track activity across all connected cameras in real-time.</p>
                
                <div className="camera-grid">
                    {cameras.map((camera) => (
                        <div key={camera.id} className={`camera-item ${camera.status.toLowerCase()}`}>
                            <div className="camera-header" onClick={() => expandCamera(camera.id)}>
                                <h3>{camera.name}</h3>
                                {expandedCamera === camera.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            </div>
                            
                            {errorMessages[camera.id] ? (
                                <div className="camera-error">
                                    <p>Error loading feed. <button onClick={() => handleRetry(camera.id)}>Retry</button></p>
                                </div>
                            ) : camera.status.toLowerCase() === 'online' ? (
                                <div className="camera-feed">
                                    <video
                                        ref={ref => webcamRefs.current[camera.id] = ref}
                                        src={camera.streamUrl}
                                        autoPlay
                                        playsInline
                                        muted
                                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                        onError={() => setErrorMessages((prev) => ({ ...prev, [camera.id]: 'Failed to load feed' }))}
                                    />
                                    <div className="live-indicator">LIVE</div>
                                </div>
                            ) : (
                                <div className="camera-offline">Camera Offline</div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default CamViewer;
