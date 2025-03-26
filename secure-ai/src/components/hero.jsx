import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import heroImage from '../assets/test image.jpg';
import './styling/hero.css';

const Hero = () => {
    const heroRef = useRef(null);
    const textRef = useRef(null);
    const imageRef = useRef(null);
    const buttonsRef = useRef(null);
    
    useGSAP(() => {
        const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
        
        tl.fromTo(textRef.current, 
            { opacity: 0, y: 50 }, 
            { 
                opacity: 1, 
                y: 0, 
                duration: 1,
                stagger: 0.2 
            }
        )
        .fromTo(imageRef.current, 
            { opacity: 0, scale: 0.9 }, 
            { 
                opacity: 1, 
                scale: 1, 
                duration: 1 
            }, 
            "-=0.5"
        )
        .fromTo(buttonsRef.current, 
            { opacity: 0, y: 30 }, 
            { 
                opacity: 1, 
                y: 0, 
                duration: 0.8 
            }, 
            "-=0.5"
        );
    }, { scope: heroRef });
    
    return(
        <div className="hero-container" id="home">
            <div className="hero-wrapper">
                <div className="hero-content" ref={heroRef}>
                    <div className="hero-text" ref={textRef}>
                        <h1>Smart Security, <span className="text-gradient">Smarter Surveillance</span></h1>
                        <p>
                            Welcome to Secure-AI, where advanced technology meets effortless security. We specialize in providing innovative solutions for seamless CCTV camera monitoring with AI-powered face tracking.
                        </p>
                        <div className="hero-buttons" ref={buttonsRef}>
                            <button className="primary-btn">Get Started</button>
                            <button className="secondary-btn">Learn More</button>
                        </div>
                    </div>
                    <div className="hero-image" ref={imageRef}>
                        <div className="image-container">
                            <div className="image-overlay"></div>
                            <img src={heroImage} alt="AI Security Camera System" />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Hero;