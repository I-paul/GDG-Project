import React, { useRef, useState, useEffect, useCallback } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import { AlertCircle, CheckCircle, Camera, Edit, Trash2 } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { db } from '../firebase';
import { collection, addDoc, getDocs, deleteDoc, doc, getDoc, updateDoc, query, where } from 'firebase/firestore';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import './styling/addcam.css';

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
        location: ""
    });
    
    useEffect(() => {
        gsap.registerPlugin(ScrollTrigger);
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
    
    // Fetch cameras function - extracted for reusability
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
                            return null; // Return null for failed fetches
                        })
                    );
                    
                    const cameraDocs = await Promise.all(cameraPromises);
                    const cameraData = cameraDocs
                        .filter(doc => doc && doc.exists())
                        .map(doc => ({ id: doc.id, ...doc.data() }));
                    
                    setCameras(cameraData);
                } else {
                    setCameras([]);
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

    const validateCameraForm = () => {
        if (!newCamera.name.trim()) {
            setError("Camera name is required.");
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

    const handleFormSubmit = async (e) => {
        e.preventDefault();
        
        if (!userId) {
            setError(ERROR_MESSAGES.NEED_LOGIN);
            navigate('/login');
            return;
        }
        
        if (!validateCameraForm()) {
            return;
        }
        
        setLoading(true);
        setError('');
        
        try {
            // Check if camera already exists
            const cameraQuery = await getDocs(
                query(collection(db, "Cameras"), where("ip", "==", newCamera.ip))
            );
            
            if (!cameraQuery.empty) {
                const existingCamera = cameraQuery.docs[0];
                const cameraData = existingCamera.data();
                
                if (cameraData.owner === userId || (cameraData.allowed_users && cameraData.allowed_users.includes(userId))) {
                    setError(ERROR_MESSAGES.ALREADY_CONNECTED);
                    setLoading(false);
                    return;
                } else {
                    setError(ERROR_MESSAGES.NO_ACCESS);
                    setLoading(false);
                    return;
                }
            }
            
            // Add the new camera
            const newCameraRef = await addDoc(collection(db, "Cameras"), {
                name: newCamera.name,
                ip: newCamera.ip,
                port: newCamera.port,
                username: newCamera.username,
                location: newCamera.location,
                owner: userId,
                allowed_users: [],
                status: "connected",
                created_at: new Date(),
                last_updated: new Date()
            });
            
            const newCameraId = newCameraRef.id;
            
            // Update user's connected cameras
            const userRef = doc(db, "Users", userId);
            const userDoc = await getDoc(userRef);
            
            if (userDoc.exists()) {
                const userData = userDoc.data();
                const connectedCameras = userData.connected_cameras || [];
                
                await updateDoc(userRef, {
                    connected_cameras: [...connectedCameras, newCameraId],
                    last_updated: new Date()
                });
            } else {
                await updateDoc(userRef, {
                    connected_cameras: [newCameraId],
                    last_updated: new Date()
                });
            }
            
            // Update local state
            setCameras([
                ...cameras, 
                { 
                    id: newCameraId, 
                    ...newCamera, 
                    status: "connected", 
                    owner: userId 
                }
            ]);
            
            // Reset form
            setNewCamera({ name: "", ip: "", port: "", username: "", location: "" });
            
            // Show success animation
            if (formRef.current) {
                gsap.fromTo(
                    formRef.current,
                    { borderColor: "rgba(76, 175, 80, 0.7)" },
                    { 
                        borderColor: "rgba(255, 105, 180, 0.2)",
                        duration: 2,
                        ease: "power2.out"
                    }
                );
            }
            
        } catch (error) {
            console.error("Error adding camera:", error);
            setError(ERROR_MESSAGES.GENERAL_ERROR);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteCamera = async (cameraId) => {
        if (isDeleting) return;
        
        if (window.confirm("Are you sure you want to remove this camera?")) {
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
                    
                    await updateDoc(userRef, {
                        connected_cameras: updatedCameras,
                        last_updated: new Date()
                    });
                    
                    // Update local state
                    setCameras(cameras.filter(camera => camera.id !== cameraId));
                }
            } catch (error) {
                console.error("Error deleting camera:", error);
                setError(ERROR_MESSAGES.DELETE_ERROR);
            } finally {
                setIsDeleting(false);
            }
        }
    };

    return (
        <section id='monitor' ref={addCamRef} className='monitor-sec'>
            <div className="monitor-container">
                <h2 className="section-title">Camera <span className="highlight">Management</span></h2>
                
                {error && (
                    <div className="error-message">
                        <AlertCircle size={18} />
                        <span>{error}</span>
                    </div>
                )}
                
                <div className="monitor-content">
                    <div className="camera-list" ref={cameraListRef}>
                        <h3>Connected Cameras</h3>
                        <div className="camera-list-container">
                            {loading ? (
                                <div className="loading-spinner"></div>
                            ) : cameras.length === 0 ? (
                                <div className="no-cameras">
                                    <Camera size={32} className="no-cameras-icon" />
                                    <p>No cameras connected yet.</p>
                                    <p>Add your first camera using the form.</p>
                                </div>
                            ) : (
                                cameras.map(camera => (
                                    <div key={camera.id} className={`camera-item ${camera.status.toLowerCase()}`}>
                                        <div className="camera-info">
                                            <h4>{camera.name}</h4>
                                            <span className="status-indicator"></span>
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
                    </div>
                    
                    <div className="add-camera-form" ref={formRef}>
                        <h3>Add New Camera</h3>
                        <form onSubmit={handleFormSubmit}>
                            <div className="form-group">
                                <label htmlFor="cameraName">Camera Name</label>
                                <input 
                                    type="text" 
                                    id="cameraName" 
                                    placeholder="My Camera" 
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
                                <label htmlFor="cameraLocation">Location</label>
                                <input 
                                    type="text" 
                                    id="cameraLocation" 
                                    placeholder="Living Room" 
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