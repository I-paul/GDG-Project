import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import './styling/addcam.css';

const AddCam = () => {
    const addCamRef = useRef(null);
    const formRef = useRef(null);
    const cameraListRef = useRef(null);

    useGSAP(() => {
        // Initial animation for camera list
        gsap.from(cameraListRef.current, {
            opacity: 0,
            x: -30,
            duration: 0.8,
            ease: "power3.out"
        });

        // Initial animation for form
        gsap.from(formRef.current, {
            opacity: 0,
            x: 30,
            duration: 0.8,
            ease: "power3.out",
            delay: 0.2
        });
    }, { scope: addCamRef });

    // Sample camera data - in a real app, this would come from props or state
    const cameras = [
        { id: 1, name: "Front Door Camera", status: "Online" },
        { id: 2, name: "Back Yard Camera", status: "Offline" },
        { id: 3, name: "Living Room Camera", status: "Online" },
        { id: 4, name: "Garage Camera", status: "Online" },
        { id: 5, name: "Side Gate Camera", status: "Offline" }
    ];

    return (
        <section id='monitor' ref={addCamRef}>
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
                                        <button className="view-btn">View</button>
                                        <button className="edit-btn">Edit</button>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                    
                    <div className="add-camera-form" ref={formRef}>
                        <h3>Add New Camera</h3>
                        <form>
                            <div className="form-group">
                                <label htmlFor="cameraName">Camera Name</label>
                                <input type="text" id="cameraName" placeholder="Enter camera name" />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraIP">IP Address</label>
                                <input type="text" id="cameraIP" placeholder="192.168.1.100" />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraPort">Port</label>
                                <input type="text" id="cameraPort" placeholder="8080" />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraUsername">Username</label>
                                <input type="text" id="cameraUsername" placeholder="admin" />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraPassword">Password</label>
                                <input type="password" id="cameraPassword" placeholder="••••••••" />
                            </div>
                            
                            <div className="form-group">
                                <label htmlFor="cameraLocation">Location</label>
                                <input type="text" id="cameraLocation" placeholder="Front Door" />
                            </div>
                            
                            <button type="submit" className="primary-btn">Add Camera</button>
                        </form>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default AddCam;