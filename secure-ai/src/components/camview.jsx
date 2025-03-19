import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import { useRef, useState, useEffect } from 'react';
import React from 'react';
import './styling/camview.css';

const CamViewer = () => {
    const viewerRef = useRef();
    const [cameras, setCameras] = useState([
        { id: 1, name: 'Front Door', status: 'online' },
        { id: 2, name: 'Back Yard', status: 'online' },
        { id: 3, name: 'Garage', status: 'online' },
        { id: 4, name: 'Living Room', status: 'offline' }
    ]);
    const [expandedCamera, setExpandedCamera] = useState(null);
    
    gsap.registerPlugin(useGSAP, ScrollTrigger);
    
    // Function to handle camera expansion
    const expandCamera = (cameraId) => {
        if (expandedCamera === cameraId) {
            setExpandedCamera(null); // Collapse if already expanded
        } else {
            setExpandedCamera(cameraId); // Expand clicked camera
        }
    };
    
    useGSAP(() => {
            gsap.fromTo(
                viewerRef.current,
                { 
                    x: '100%',  
                    opacity: 0, 
                    duration: 1.3,
                    ease: 'power2.out'
                },
                { 
                    x: 0,  
                    opacity: 1, 
                    duration: 1.3,  
                    ease: 'power2.out',
                    scrollTrigger: {
                        trigger: viewerRef.current,
                        start: 'top bottom', 
                        end: 'top 50%',       
                        scrub: 1,             
                        toggleActions: 'play reverse play reverse',  
                        pinSpacing: false,   
                        markers: false        
                    }
                }
            );
        // Animate camera grid items
        gsap.fromTo(
            '.camera-item',
            { 
                y: 50, 
                opacity: 0 
            },
            { 
                y: 0, 
                opacity: 1, 
                duration: 1.6, 
                stagger: 0.2,
                ease: 'back.out(1.2)',
                scrollTrigger: {
                    trigger: '.camera-grid',
                    start: 'top 70%',
                    toggleActions: 'play none none reverse'
                }
            }
        );
    });
    
    useEffect(() => {
        if (expandedCamera) {
            document.body.style.overflow = 'hidden'; // Prevent scrolling when expanded
            gsap.to('.expanded-overlay', { 
                opacity: 1, 
                duration: 0.3, 
                ease: 'power2.inOut' 
            });
            gsap.fromTo(
                '.expanded-camera', 
                { scale: 0.8, opacity: 0 }, 
                { scale: 1, opacity: 1, duration: 0.5, ease: 'back.out(1.7)' }
            );
        } else {
            document.body.style.overflow = 'auto'; // Restore scrolling
        }
        
        // Cleanup function
        return () => {
            document.body.style.overflow = 'auto';
        };
    }, [expandedCamera]);
    
    return (
        <section id="monitor" className='viewer' ref={viewerRef}>
            <div className="viewer-container">
                <div className="viewer-header">
                    <h2>Live Camera <span className="highlight">Monitoring</span></h2>
                    <p>View and track activity across all your connected cameras in real-time.</p>
                </div>
                
                <div className="camera-grid">
                    {cameras.map((camera) => (
                        <div 
                            key={camera.id} 
                            className={`camera-item ${camera.status}`}
                            onClick={() => camera.status === 'online' && expandCamera(camera.id)}
                        >
                            <div className="camera-feed">
                                {/* Placeholder for actual camera feed */}
                                <div className="camera-placeholder">
                                    {camera.status === 'online' ? (
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
                                <div className={`status-indicator ${camera.status}`}>
                                    <span className="status-dot"></span>
                                    <span className="status-text">{camera.status}</span>
                                </div>
                            </div>
                            
                            {camera.status === 'online' && (
                                <div className="camera-controls">
                                    <button className="control-btn expand-btn">
                                        <span className="icon">⤢</span>
                                    </button>
                                    <button className="control-btn track-btn">
                                        <span className="icon">👤</span>
                                    </button>
                                </div>
                            )}
                        </div>
                    ))}
                </div>
                
                <div className="viewer-footer">
                    <p>Connected Cameras: <span className="connected-count">{cameras.filter(c => c.status === 'online').length}/{cameras.length}</span></p>
                    <button className="add-camera-btn">
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
                                <span className="icon">👤</span> Track Person
                            </button>
                            <button className="expanded-control-btn">
                                <span className="icon">📷</span> Screenshot
                            </button>
                            <button className="expanded-control-btn">
                                <span className="icon">⚙️</span> Settings
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </section>
    );
};

export default CamViewer;