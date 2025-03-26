import React, { useRef, useState, useEffect } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import { ArrowLeft, Camera, UserCheck, Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './styling/camview.css';
import Hls from 'hls.js';

const CamViewer = () => {
    const viewerRef = useRef();
    const navigate = useNavigate();
    const [cameras, setCameras] = useState([]);
    const [expandedCamera, setExpandedCamera] = useState(null);
    const [openDropdowns, setOpenDropdowns] = useState({});
    const webcamRefs = useRef({});
    const expandedWebcamRef = useRef(null);
    
    gsap.registerPlugin(useGSAP, ScrollTrigger);
    
    // Load cameras from localStorage on component mount
    useEffect(() => {
        const storedCameras = localStorage.getItem('cameras');
        if (storedCameras) {
            // Parse stored cameras and ensure they have a type property
            const parsedCameras = JSON.parse(storedCameras).slice(0, 5).map(cam => ({
                ...cam,
                type: cam.type || 'local' // Default to local if not specified
            }));
            setCameras(parsedCameras);
        } else {
            // Default cameras with type property
            setCameras([
                { 
                    id: 1, 
                    name: 'Front Door Camera', 
                    status: 'Online', 
                    location: 'Front Door', 
                    type: 'external', 
                    streamUrl: 'http://your-streaming-server.com/streams/back-door.m3u8',
                    streamFormat: 'hls'
                },
                { 
                    id: 2, 
                    name: 'Back Door Camera', 
                    status: 'Online', 
                    location: 'Back Door', 
                    type: 'external', 
                    streamUrl: 'http://your-streaming-server.com/streams/back-door.m3u8',
                    streamFormat: 'hls'
                },
                { 
                    id: 3, 
                    name: 'Garage Camera', 
                    status: 'Offline', 
                    location: 'Garage', 
                    type: 'external', 
                    streamUrl: 'http://your-streaming-server.com/streams/garage.m3u8',
                    streamFormat: 'hls'
                },
                { 
                    id: 4, 
                    name: 'Local Camera', 
                    status: 'Online', 
                    location: 'Living Room', 
                    type: 'local'
                },
            ]);
        }
    }, []);

    // Initialize webcam for each online camera
    useEffect(() => {
        // Clean up previous webcam streams
        Object.values(webcamRefs.current).forEach(videoRef => {
            if (videoRef && videoRef.srcObject) {
                const tracks = videoRef.srcObject.getTracks();
                tracks.forEach(track => track.stop());
            }
        });

        // Initialize webcams for online cameras
        cameras.forEach(camera => {
            if (camera.status.toLowerCase() === 'online') {
                initializeWebcam(camera.id);
            }
        });

        // Cleanup function to stop all webcam streams when component unmounts
        return () => {
            Object.values(webcamRefs.current).forEach(videoRef => {
                if (videoRef) {
                    if (videoRef.srcObject) {
                        // For local webcams
                        const tracks = videoRef.srcObject.getTracks();
                        tracks.forEach(track => track.stop());
                    }
                    // For external cameras, stopping playback
                    videoRef.pause();
                    videoRef.src = '';
                }
            });
            
            if (expandedWebcamRef.current) {
                if (expandedWebcamRef.current.srcObject) {
                    // For local webcams
                    const tracks = expandedWebcamRef.current.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                }
                // For external cameras
                expandedWebcamRef.current.pause();
                expandedWebcamRef.current.src = '';
            }
        };
    }, [cameras]);

    // Initialize webcam for expanded view when camera is expanded
    useEffect(() => {
        let timeoutId;
        if (expandedCamera) {
            // Wait a moment for the DOM to update
            timeoutId = setTimeout(() => {
                initializeExpandedWebcam(expandedCamera);
            }, 100);
        }
        
        // Clean up the timeout if component unmounts or expandedCamera changes
        return () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
            
            Object.values(webcamRefs.current).forEach(videoRef => {
                if (videoRef) {
                    if (videoRef.srcObject) {
                        // For local webcams
                        const tracks = videoRef.srcObject.getTracks();
                        tracks.forEach(track => track.stop());
                    }
                    // For external cameras, stopping playback
                    videoRef.pause();
                    videoRef.src = '';
                }
            });
            
            if (expandedWebcamRef.current) {
                if (expandedWebcamRef.current.srcObject) {
                    // For local webcams
                    const tracks = expandedWebcamRef.current.srcObject.getTracks();
                    tracks.forEach(track => track.stop());
                }
                // For external cameras
                expandedWebcamRef.current.pause();
                expandedWebcamRef.current.src = '';
            }
        };
    }, [expandedCamera]); // Keep this dependency array!

    // Function to initialize webcam for a specific camera
    const initializeWebcam = async (cameraId) => {
        const camera = cameras.find(c => c.id === cameraId);
        if (!camera || camera.status.toLowerCase() !== 'online') return;
        
        try {
            if (camera.type === 'external' && camera.streamUrl) {
                // For external cameras with stream URLs
                if (webcamRefs.current[cameraId]) {
                    const videoElement = webcamRefs.current[cameraId];
                    
                    // Handle HLS streams (common format after RTSP conversion)
                    if (camera.streamUrl.includes('.m3u8') && Hls.isSupported()) {
                        const hls = new Hls();
                        hls.loadSource(camera.streamUrl);
                        hls.attachMedia(videoElement);
                        hls.on(Hls.Events.MANIFEST_PARSED, () => {
                            videoElement.play();
                        });
                    } 
                    // Handle direct mp4 streams or other formats browsers can play natively
                    else if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
                        // For Safari which has native HLS support
                        videoElement.src = camera.streamUrl;
                        videoElement.addEventListener('loadedmetadata', () => {
                            videoElement.play();
                        });
                    }
                    // Handle MJPEG streams
                    else if (camera.streamUrl.includes('mjpeg') || camera.streamFormat === 'mjpeg') {
                        // For MJPEG streams, we use img tag instead of video
                        // This requires modifying your JSX rendering for this case
                        console.log('MJPEG stream detected');
                    } 
                    // Fallback for other formats
                    else {
                        videoElement.src = camera.streamUrl;
                        videoElement.play().catch(e => console.error('Error playing video:', e));
                    }
                }
            } else {
                // For local webcam (existing implementation)
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 1280 },
                        height: { ideal: 720 }
                    } 
                });
                
                if (webcamRefs.current[cameraId] && webcamRefs.current[cameraId] !== null) {
                    webcamRefs.current[cameraId].srcObject = stream;
                    webcamRefs.current[cameraId].play();
                }
            }
        } catch (error) {
            console.error(`Error accessing camera ${cameraId}:`, error);
        }
    };

    // Function to initialize webcam for expanded view
    const initializeExpandedWebcam = async (cameraId) => {
        const camera = cameras.find(c => c.id === cameraId);
        if (!camera || camera.status.toLowerCase() !== 'online') return;
        
        try {
            if (camera.type === 'external' && camera.streamUrl) {
                // For external cameras with stream URLs in expanded view
                if (expandedWebcamRef.current) {
                    const videoElement = expandedWebcamRef.current;
                    
                    // Handle HLS streams
                    if (camera.streamUrl.includes('.m3u8') && Hls.isSupported()) {
                        const hls = new Hls();
                        hls.loadSource(camera.streamUrl);
                        hls.attachMedia(videoElement);
                        hls.on(Hls.Events.MANIFEST_PARSED, () => {
                            videoElement.play();
                        });
                    } 
                    // For Safari with native HLS support
                    else if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
                        videoElement.src = camera.streamUrl;
                        videoElement.addEventListener('loadedmetadata', () => {
                            videoElement.play();
                        });
                    }
                    // Fallback for other formats
                    else {
                        videoElement.src = camera.streamUrl;
                        videoElement.play().catch(e => console.error('Error playing expanded video:', e));
                    }
                }
            } else {
                // For local webcam (existing implementation)
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 1920 },
                        height: { ideal: 1080 }
                    } 
                });
                
                if (expandedWebcamRef.current && expandedWebcamRef.current !== null) {
                    expandedWebcamRef.current.srcObject = stream;
                    expandedWebcamRef.current.play();
                }
            }
        } catch (error) {
            console.error(`Error accessing webcam for expanded view:`, error);
        }
    };
    
    // Function to handle camera expansion
    const expandCamera = (cameraId) => {
        const camera = cameras.find(c => c.id === cameraId);
        if (camera && camera.status.toLowerCase() === 'online') {
            if (expandedCamera === cameraId) {
                setExpandedCamera(null); // Collapse if already expanded
            } else {
                setExpandedCamera(cameraId); // Expand clicked camera
            }
        }
    };
    
    // Function to toggle dropdown
    const toggleDropdown = (e, cameraId) => {
        e.stopPropagation(); // Prevent camera expansion when clicking dropdown
        setOpenDropdowns(prev => ({
            ...prev,
            [cameraId]: !prev[cameraId]
        }));
    };
    
    // Navigate back to add camera page
    const handleBackToManagement = () => {
        navigate('/');
    };
    
    // Navigate to add camera form
    const handleAddCamera = () => {
        navigate('/');
    };

    // Function to take a screenshot
    const takeScreenshot = () => {
        if (expandedWebcamRef.current) {
            // Create a canvas element
            const canvas = document.createElement('canvas');
            canvas.width = expandedWebcamRef.current.videoWidth;
            canvas.height = expandedWebcamRef.current.videoHeight;
            
            // Draw the current video frame on the canvas
            const ctx = canvas.getContext('2d');
            ctx.drawImage(expandedWebcamRef.current, 0, 0, canvas.width, canvas.height);
            
            // Convert the canvas to a data URL and create a download link
            const dataURL = canvas.toDataURL('image/png');
            const a = document.createElement('a');
            a.href = dataURL;
            a.download = `camera-screenshot-${Date.now()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    };
    
    // Determine class name for camera grid based on number of cameras
    const getCameraGridClassName = () => {
        if (cameras.length === 1) {
            return "camera-grid single-camera";
        } else if (cameras.length <= 4) {
            return "camera-grid multi-camera";
        } else {
            return "camera-grid many-cameras";
        }
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
                
                <div className={getCameraGridClassName()}>
                    {cameras.map((camera) => (
                        <div 
                            key={camera.id} 
                            className={`camera-item ${camera.status.toLowerCase()}`}
                            onClick={() => expandCamera(camera.id)}
                        >
                            {/* Camera feed container */}
                            <div className="camera-feed-fullscreen">
                                {/* Placeholder for actual camera feed */}
                                <div className="camera-placeholder">
                                    {camera.status.toLowerCase() === 'online' ? (
                                        <>
                                            {/* Webcam Video Element */}
                                            <video
                                                ref={ref => webcamRefs.current[camera.id] = ref}
                                                autoPlay
                                                playsInline
                                                muted
                                                // For external cameras, we need src instead of srcObject
                                                // Don't set src here, it will be set in initializeWebcam
                                                style={{
                                                    width: '100%',
                                                    height: '100%',
                                                    objectFit: 'cover',
                                                    position: 'absolute',
                                                    top: 0,
                                                    left: 0,
                                                    zIndex: 1
                                                }}
                                            />
                                            <div className="placeholder-icon" style={{ opacity: 0 }}></div>
                                            <div className="live-indicator">LIVE</div>
                                        </>
                                    ) : (
                                        <div className="offline-message">Camera Offline</div>
                                    )}
                                </div>
                                
                                {/* Overlay with camera info */}
                                <div className="camera-overlay">
                                    <div className="camera-info-bar">
                                        <div className="camera-title">
                                            <h3>{camera.name}</h3>
                                        </div>
                                        <button 
                                            className="dropdown-toggle-btn"
                                            onClick={(e) => toggleDropdown(e, camera.id)}
                                        >
                                            {openDropdowns[camera.id] ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                        </button>
                                    </div>
                                    
                                    {openDropdowns[camera.id] && (
                                        <div className="dropdown-panel">
                                            <div className="detailed-info">
                                                <div className="info-row">
                                                    <span className="info-label">Location:</span>
                                                    <span className="info-value">{camera.location}</span>
                                                </div>
                                                <div className="info-row">
                                                    <span className="info-label">Status:</span>
                                                    <span className={`info-value status-${camera.status.toLowerCase()}`}>
                                                        {camera.status}
                                                    </span>
                                                </div>
                                                <div className="info-row">
                                                    <span className="info-label">Camera ID:</span>
                                                    <span className="info-value">CAM-{camera.id.toString().padStart(4, '0')}</span>
                                                </div>
                                                <div className="info-row">
                                                    <span className="info-label">Last Check:</span>
                                                    <span className="info-value">Just now</span>
                                                </div>
                                            </div>
                                            
                                            {camera.status.toLowerCase() === 'online' && (
                                                <div className="camera-controls">
                                                    <button className="control-btn expand-btn" onClick={(e) => {
                                                        e.stopPropagation();
                                                        expandCamera(camera.id);
                                                    }}>
                                                        <span className="icon">⤢</span>
                                                    </button>
                                                    <button className="control-btn track-btn" onClick={(e) => e.stopPropagation()}>
                                                        <UserCheck size={16} />
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
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
                            {/* Webcam for expanded view */}
                            <video
                                ref={expandedWebcamRef}
                                autoPlay
                                playsInline
                                muted
                                // For external cameras, we need src instead of srcObject
                                // Don't set src here, it will be set in initializeExpandedWebcam
                                style={{
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'cover',
                                    position: 'absolute',
                                    top: 0,
                                    left: 0
                                }}
                            />
                            <div className="expanded-placeholder" style={{ zIndex: -1 }}>
                                <div className="placeholder-icon large"></div>
                            </div>
                            <div className="live-indicator">LIVE</div>
                        </div>
                        <div className="expanded-controls">
                            <button className="expanded-control-btn">
                                <UserCheck size={18} /> Track Person
                            </button>
                            <button className="expanded-control-btn" onClick={takeScreenshot}>
                                <Camera size={18} /> Screenshot
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