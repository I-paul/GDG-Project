import React, { useRef, useState } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import { AlertCircle, CheckCircle, Camera, Edit } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './styling/addcam.css';

const AddCam = () => {
    const addCamRef = useRef(null);
    const formRef = useRef(null);
    const cameraListRef = useRef(null);
    const navigate = useNavigate();
    
    // State for camera management
    const [cameras, setCameras] = useState([
        { id: 1, name: "Front Door Camera", status: "Online", ip: "192.168.1.101", port: "8080", username: "admin", location: "Front Door" },
        { id: 2, name: "Back Yard Camera", status: "Offline", ip: "192.168.1.102", port: "8080", username: "admin", location: "Back Yard" },
        { id: 3, name: "Living Room Camera", status: "Online", ip: "192.168.1.103", port: "8080", username: "admin", location: "Living Room" },
        { id: 4, name: "Garage Camera", status: "Online", ip: "192.168.1.104", port: "8080", username: "admin", location: "Garage" },
        { id: 5, name: "Side Gate Camera", status: "Offline", ip: "192.168.1.105", port: "8080", username: "admin", location: "Side Gate" }
    ]);
    
    // State for new camera form
    const [newCamera, setNewCamera] = useState({
        name: "",
        ip: "",
        port: "",
        username: "",
        password: "",
        location: ""
    });
    
    // Initialize GSAP
    gsap.registerPlugin(useGSAP, ScrollTrigger);
    
    useGSAP(() => {
        gsap.fromTo(cameraListRef.current, { opacity: 0 }, {
            opacity: 1,
            duration: 2,
            ease: "power3.out",
            scrollTrigger: {
                trigger: addCamRef.current,
                start: 'top 35%', 
                end: '+=500', 
                scrub: 1, 
                toggleActions: 'restart pause reverse reverse'
            },
            delay: 0.4
        });
        
        gsap.fromTo(formRef.current, { opacity: 0 }, {
            opacity: 1,
            duration: 2,
            ease: "power3.out",
            scrollTrigger: {
                trigger: addCamRef.current,
                start: 'top 35%', 
                end: '+=500', 
                scrub: 1, 
                toggleActions: 'restart pause reverse reverse'
            },
            delay: 0.4
        });
    }, { scope: addCamRef });

    // Handle form input changes
    const handleInputChange = (e) => {
        const { id, value } = e.target;
        setNewCamera({
            ...newCamera,
            [id.replace('camera', '').toLowerCase()]: value
        });
    };

    // Handle form submission
    const handleFormSubmit = (e) => {
        e.preventDefault();
        
        // Create new camera object with additional properties required by camview.jsx
        const newCameraObj = {
            id: cameras.length + 1,
            name: newCamera.name,
            status: "Online", // Default status
            ip: newCamera.ip,
            port: newCamera.port,
            username: newCamera.username,
            location: newCamera.location,
            // Add these required properties for camview.jsx
            type: "external", // Set the camera type to 'external'
            streamUrl: `http://${newCamera.ip}:${newCamera.port}/stream`, // Construct a stream URL from IP and port
            streamFormat: "hls" // Default to HLS format which is common for IP cameras
        };
        
        // Add to cameras array
        setCameras([...cameras, newCameraObj]);
        
        // Store updated cameras in localStorage
        localStorage.setItem('cameras', JSON.stringify([...cameras, newCameraObj]));
        
        // Reset form
        setNewCamera({
            name: "",
            ip: "",
            port: "",
            username: "",
            password: "",
            location: ""
        });
        
        // Success animation
        const formElement = formRef.current;
        gsap.to(formElement, {
            borderColor: "var(--purple)",
            boxShadow: "0 0 20px var(--purple-transparent)",
            duration: 0.3,
            onComplete: () => {
                gsap.to(formElement, {
                    borderColor: "rgba(255, 105, 180, 0.2)",
                    boxShadow: "0 8px 32px rgba(255, 105, 180, 0.1)",
                    duration: 0.5,
                    delay: 1
                });
            }
        });
    };

    // Navigate to camera view page
    const handleViewCameras = () => {
        navigate('/camera-view');
    };
    
    // Handle editing camera
    const handleEditCamera = (cameraId) => {
        // You can implement the edit functionality here
        // For now, just navigate to camera view
        navigate('/camera-view');
    };

    return (
        <section id='monitor' ref={addCamRef} className='monitor-sec'>
            <div className="monitor-container">
                <h2 className="section-title">Camera <span className="highlight">Management</span></h2>
                
                <div className="monitor-content">
                    <div className="camera-list" ref={cameraListRef}>
                        <h3>Connected Cameras</h3>
                        <div className="camera-list-container">
                            {cameras.map(camera => (
                                <div key={camera.id} className={`camera-item ${camera.status.toLowerCase()}`}>
                                    <div className="camera-info">
                                        <h4>{camera.name}</h4>
                                        <span className="status-indicator"></span>
                                        <span className="status-text">{camera.status}</span>
                                    </div>
                                    <div className="camera-actions">
                                        <button 
                                            className="edit-btn"
                                            onClick={() => handleEditCamera(camera.id)}
                                        >
                                            <Edit size={16} />
                                        </button>
                                    </div>
                                </div>
                            ))}
                        </div>
                        
                        <div className="view-all-section">
                            <button 
                                className="view-all-btn"
                                onClick={handleViewCameras}
                                disabled={!cameras.some(cam => cam.status === "Online")}
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
                                    placeholder="••••••••"
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
                            
                            <button type="submit" className="primary-btn">
                                <CheckCircle size={18} className="btn-icon" />
                                Add Camera
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default AddCam;