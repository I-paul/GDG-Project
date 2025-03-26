import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import { db } from '../firebase';
import { collection, addDoc, getDocs, deleteDoc, doc, getDoc, updateDoc, query, where } from 'firebase/firestore';
import { CheckCircle, Edit, Trash2, AlertCircle, Info, Camera } from 'lucide-react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import './styling/addcam.css';

gsap.registerPlugin(ScrollTrigger);

// Error messages object for better management
const ERROR_MESSAGES = {
  NO_ACCESS: "You don't have access to this camera.",
  ALREADY_CONNECTED: "You are already connected to this camera.",
  NEED_LOGIN: "You must be logged in to add a camera.",
  GENERAL_ERROR: "Error occurred. Please try again.",
  FETCH_ERROR: "Error fetching cameras. Please refresh the page.",
  DELETE_ERROR: "Error removing camera. Please try again."
};

// Input validation functions
const validateIpAddress = (ip) => {
  const ipPattern = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;
  return ipPattern.test(ip);
};

const validatePort = (port) => {
  return /^\d+$/.test(port) && parseInt(port) >= 0 && parseInt(port) <= 65535;
};

const AddCam = () => {
    const addCamRef = useRef(null);
    const formRef = useRef(null);
    const cameraListRef = useRef(null);
    
    const navigate = useNavigate();
    
    const auth = getAuth();
    const [userId, setUserId] = useState(null);
    const [cameras, setCameras] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [isDeleting, setIsDeleting] = useState(false);
    const [newCamera, setNewCamera] = useState({
        name: "",
        ip: "",
        port: "",
        username: "",
        password: "", 
        location: ""
    });
    
    useEffect(() => {
        if (!addCamRef.current || !formRef.current || !cameraListRef.current) {
            console.warn("One or more animation targets are missing.");
            return;
        }

        // Animation for the camera list
        gsap.fromTo(
            cameraListRef.current,
            { opacity: 0, y: 100, scale: 0.9 },
            {
                opacity: 1,
                y: 0,
                scale: 1,
                duration: 1.2,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: addCamRef.current,
                    start: 'top 70%',
                    toggleActions: 'play none none reverse',
                },
            }
        );

        // Animation for the form
        gsap.fromTo(
            formRef.current,
            { opacity: 0, x: -50, rotate: -10 },
            {
                opacity: 1,
                x: 0,
                rotate: 0,
                duration: 1,
                ease: 'back.out(1.7)',
                scrollTrigger: {
                    trigger: formRef.current,
                    start: 'top 70%',
                    toggleActions: 'play none none reverse',
                },
            }
        );
    }, []);
    
    // Check authentication status
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (user) => {
            if (user) {
                setUserId(user.uid);
            } else {
                navigate('/login');
            }
            setLoading(false);
        });
        return () => unsubscribe();
    }, [auth, navigate]);
    
    // Fetch cameras function
    const fetchCameras = useCallback(async (uid) => {
        if (!uid) return;
        
        setLoading(true);
        setError('');
        
        try {
            const userDoc = await getDoc(doc(db, "Users", uid));
            
            if (userDoc.exists()) {
                const userData = userDoc.data();
                const connectedCameraIds = userData.connected_cameras || [];
                
                if (connectedCameraIds.length > 0) {
                    const cameraPromises = connectedCameraIds.map(camId => 
                        getDoc(doc(db, "Cameras", camId))
                        .catch(err => {
                            console.error(`Error fetching camera ${camId}:`, err);
                            return null;
                        })
                    );
                    
                    const cameraDocs = await Promise.all(cameraPromises);
                    const cameraData = cameraDocs.filter(doc => doc && doc.exists())
                        .map(doc => ({ 
                            id: doc.id, 
                            ...doc.data(),
                            type: "external",
                            streamUrl: `http://${doc.data().ip}:${doc.data().port}/stream`,
                            streamFormat: "hls"
                        }));
                    
                    setCameras(cameraData);
                    
                    // Sync with localStorage for camview.jsx
                    localStorage.setItem('cameras', JSON.stringify(cameraData));
                } else {
                    setCameras([]);
                    localStorage.removeItem('cameras');
                }
            }
        } catch (error) {
            console.error("Error fetching cameras:", error);
            setError(ERROR_MESSAGES.FETCH_ERROR);
        } finally {
            setLoading(false);
        }
    }, []);
    
    // Fetch cameras when userId changes
    useEffect(() => {
        if (userId) {
            fetchCameras(userId);
        }
    }, [userId, fetchCameras]);

    // Enhanced error handling with more informative tooltips
    const renderErrorTooltip = () => {
        return error ? (
            <div className="error-tooltip">
                <AlertCircle size={18} className="error-icon" />
                <span className="error-text">{error}</span>
                <div className="error-info-tooltip">
                    <Info size={14} />
                    <span className="tooltip-content">
                        {error === ERROR_MESSAGES.NO_ACCESS 
                            ? "This camera might be owned by another user or requires special permissions." 
                            : "Please check your input and try again."}
                    </span>
                </div>
            </div>
        ) : null;
    };

    // Optional: Add camera status color mapping
    const getCameraStatusColor = (status) => {
        const statusColors = {
            'Online': '#4CAF50',
            'Offline': '#F44336',
            'Connecting': '#FFC107'
        };
        return statusColors[status] || '#A0A0A0';
    };

    // Handle input changes
    const handleInputChange = (e) => {
        const { id, value } = e.target;
        const fieldName = id.replace('camera', '').toLowerCase();
        
        setNewCamera({
            ...newCamera,
            [fieldName]: value
        });
        
        // Clear error when user types
        if (error) setError('');
    };

    // Validate camera form
    const validateCameraForm = () => {
        if (!newCamera.name.trim()) {
            setError("Camera name is required.");
            return false;
        }
        if (!newCamera.password.trim()) {
            setError("Camera password is required.");
            return false;
        }
        if (!validateIpAddress(newCamera.ip)) {
            setError("Please enter a valid IP address.");
            return false;
        }
        
        if (!validatePort(newCamera.port)) {
            setError("Please enter a valid port number (0-65535).");
            return false;
        }
        
        return true;
    };

    // Handle form submission
    const handleFormSubmit = async (e) => {
        e.preventDefault();
        
        if (!validateCameraForm()) {
            return;
        }
        
        try {
            // Add the new camera with password
            const newCameraRef = await addDoc(collection(db, "Cameras"), {
                name: newCamera.name,
                ip: newCamera.ip,
                port: newCamera.port,
                username: newCamera.username,
                password: newCamera.password, // Include password
                location: newCamera.location,
                owner: userId,
                allowed_users: [],
                status: "Online",
                type: "external", // Explicitly set type
                streamUrl: `rtsp://${newCamera.username}:${newCamera.password}@${newCamera.ip}:${newCamera.port}/stream`,
                streamFormat: "hls",
                created_at: new Date(),
                last_updated: new Date()
            });
            
            // Rest of the existing submission logic remains the same
        } catch (error) {
            console.error("Error adding camera:", error);
            setError(ERROR_MESSAGES.GENERAL_ERROR);
        }
    };

    // Handle camera deletion
    const handleDeleteCamera = async (cameraId) => {
        if (isDeleting) return;
        
        const confirmDelete = window.confirm("Are you absolutely sure you want to remove this camera? This will permanently delete the camera from your account.");
        
        if (confirmDelete) {
            setIsDeleting(true);
            
            try {
                // Get user document
                const userRef = doc(db, "Users", userId);
                const userDoc = await getDoc(userRef);
                
                if (userDoc.exists()) {
                    const userData = userDoc.data();
                    const connectedCameras = userData.connected_cameras || [];
                    
                    // Remove camera ID from user's connected_cameras array
                    const updatedCameras = connectedCameras.filter(id => id !== cameraId);
                    
                    // Update user document
                    await updateDoc(userRef, {
                        connected_cameras: updatedCameras,
                        last_updated: new Date()
                    });
                    
                    // Delete the camera document from Cameras collection
                    await deleteDoc(doc(db, "Cameras", cameraId));
                    
                    // Fetch updated cameras
                    await fetchCameras(userId);
                    
                    // Optional: Add a toast or notification for successful deletion
                    console.log("Camera successfully deleted");
                }
            } catch (error) {
                console.error("Error deleting camera:", error);
                setError(ERROR_MESSAGES.DELETE_ERROR);
            } finally {
                setIsDeleting(false);
            }
        }
    };

    // Navigate to camera view
    const handleViewCameras = () => {
        navigate('/camera-view');
    };

    return (
        <section id='monitor' ref={addCamRef} className='monitor-sec'>
            <div className="monitor-container">
                <h2 className="section-title">Camera <span className="highlight">Management</span></h2>
                
                {renderErrorTooltip()}
                
                <div className="monitor-content">
                    <div className="camera-list" ref={cameraListRef}>
                        <h3>Connected Cameras</h3>
                        <div className="camera-list-container">
                            {loading ? (
                                <div className="loading-spinner"></div>
                            ) : cameras.length === 0 ? (
                                <div className="camera-empty-state">
                                    <Camera size={64} className="empty-state-icon" />
                                    <h3>No Cameras Added</h3>
                                    <p>Start by adding your first camera using the form on the right.</p>
                                </div>
                            ) : (
                                cameras.map(camera => (
                                    <div key={camera.id} className={`camera-item ${camera.status.toLowerCase()}`}>
                                        <div className="camera-info">
                                            <h4>{camera.name}</h4>
                                            <span 
                                                className="status-indicator" 
                                                style={{
                                                    backgroundColor: getCameraStatusColor(camera.status),
                                                    boxShadow: `0 0 10px ${getCameraStatusColor(camera.status)}`
                                                }}
                                            ></span>
                                            <span className="status-text">{camera.status}</span>
                                        </div>
                                        <div className="camera-actions">
                                            <button 
                                                className="edit-btn" 
                                                title="Edit Camera"
                                                onClick={() => navigate(`/edit-camera/${camera.id}`)}
                                            >
                                                <Edit size={16} />
                                            </button>
                                            <button 
                                                className="delete-btn" 
                                                title="Remove Camera"
                                                onClick={() => handleDeleteCamera(camera.id)}
                                                disabled={isDeleting}
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                        
                        <div className="view-all-section">
                            <button 
                                className="view-all-btn"
                                onClick={handleViewCameras}
                                disabled={cameras.length === 0}
                            >
                                <Camera size={18} />
                                View Cameras
                            </button>
                        </div>
                    </div>
                    
                    <div className="add-camera-form" ref={formRef}>
                        <h3>Add New Camera</h3>
                        <form onSubmit={handleFormSubmit}>
                            <div className="form-group">
                                <label htmlFor="cameraName">Camera Name</label>
                                <input 
                                    type="text" 
                                    id="cameraName" 
                                    placeholder="Enter camera name"
                                    value={newCamera.name}
                                    onChange={handleInputChange}
                                    required
                                />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraIP">IP Address</label>
                                <input 
                                    type="text" 
                                    id="cameraIP" 
                                    placeholder="192.168.1.100"
                                    value={newCamera.ip}
                                    onChange={handleInputChange}
                                    required
                                    pattern="^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
                                    title="Please enter a valid IP address"
                                />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraPort">Port</label>
                                <input 
                                    type="text" 
                                    id="cameraPort" 
                                    placeholder="8080"
                                    value={newCamera.port}
                                    onChange={handleInputChange}
                                    required
                                    pattern="^[0-9]{1,5}$"
                                    title="Please enter a valid port number"
                                />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraUsername">Username</label>
                                <input 
                                    type="text" 
                                    id="cameraUsername" 
                                    placeholder="admin"
                                    value={newCamera.username}
                                    onChange={handleInputChange}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="cameraPassword">Password</label>
                                <input 
                                    type="password" 
                                    id="cameraPassword" 
                                    placeholder="Camera access password"
                                    value={newCamera.password}
                                    onChange={handleInputChange}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="cameraLocation">Location</label>
                                <input 
                                    type="text" 
                                    id="cameraLocation" 
                                    placeholder="Front Door"
                                    value={newCamera.location}
                                    onChange={handleInputChange}
                                    required
                                />
                            </div>
                            
                            <button 
                                type="submit" 
                                className="primary-btn"
                                disabled={loading}
                            >
                                {loading ? (
                                    <span className="button-loader"></span>
                                ) : (
                                    <>
                                        <CheckCircle size={18} className="btn-icon" /> Add Camera
                                    </>
                                )}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default AddCam;