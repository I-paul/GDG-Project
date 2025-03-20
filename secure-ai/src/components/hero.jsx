import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import heroImage from '../assets/test image.jpg';
import './styling/hero.css';

const Hero = () => {
    const heroRef = useRef(null);
    const textRef = useRef(null);
    const buttonRef = useRef(null);
    const imageRef = useRef(null);
    
    useGSAP(() => {
        // Initial setup
        gsap.set(textRef.current, { opacity: 0, x: -50 });
        gsap.set(buttonRef.current, { opacity: 0, y: 20 });
        gsap.set(imageRef.current, { opacity: 0, scale: 0.9 });
        
        // Animate hero text and buttons on load
        const initialTl = gsap.timeline({ defaults: { ease: "power3.out" } });
        initialTl
            .to(textRef.current, { 
                opacity: 1, 
                x: 0,
                duration: 1,
            })
            .to(buttonRef.current, { 
                opacity: 1, 
                y: 0,
                duration: 0.8
            }, "-=0.4")
            .to(imageRef.current, {
                opacity: 1,
                scale: 1,
                duration: 1
            }, "-=0.6");
        
    }, { scope: heroRef });
    
    return(
        <div className="hero-container">
            <section id='home' className="hero-section" ref={heroRef}>
                <div className="hero-content">
                    <div className="hero-text" ref={textRef}>
                        <h1>Smart Security, <span className="highlight">Smarter Surveillance</span></h1>
                        <p>
                            Welcome to Secure-AI, where advanced technology meets effortless security. We specialize in providing innovative solutions for seamless CCTV camera monitoring with AI-powered face tracking.
                        </p>
                        <p>
                            Our cutting-edge technology allows you to effortlessly track individuals across multiple cameras, ensuring comprehensive surveillance and enhanced security for your property or business.
                        </p>
                        <div className="hero-actions" ref={buttonRef}>
                            <button className="primary-btn">Get Started</button>
                            <button className="secondary-btn">Learn More</button>
                        </div>
                    </div>
                </div>
                <div className="hero-image" ref={imageRef}>
                    <div className="image-container">
                        <div className="image-overlay"></div>
                        <img src={heroImage} alt="AI Security Camera System" />
                    </div>
                </div>
            </section>
        </div>
    );
};

export default Hero;