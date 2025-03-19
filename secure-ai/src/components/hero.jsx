import {useGSAP} from '@gsap/react';
import gsap from 'gsap';
import { useRef } from 'react';
import React from 'react';
import './styling/hero.css';

const Hero = () => {
    const heroRef = useRef();
    const textRef = useRef();
    const buttonRef = useRef();
    const imageRef = useRef();
    
    gsap.registerPlugin(useGSAP);
    
    useGSAP(() => {
        const tl = gsap.timeline();
        
        tl.fromTo(textRef.current, 
            { opacity: 0, y: 50 }, 
            { opacity: 1, y: 0, duration: 1.2, ease: "power3.out" }
        );
        
        tl.fromTo(buttonRef.current, 
            { opacity: 0, y: 30 }, 
            { opacity: 1, y: 0, duration: 0.8, ease: "back.out" }, 
            "-=0.6"
        );
        
        tl.fromTo(imageRef.current, 
            { opacity: 0, scale: 0.9 }, 
            { opacity: 1, scale: 1, duration: 1, ease: "power2.out" }, 
            "-=0.8"
        );
    });
    
    return(
        <section id='home' className="hero-section" ref={heroRef}>
            <div className="hero-container">
                <div className="hero-content">
                    <div className="hero-text" ref={textRef}>
                        <h1>Smart Security, <span className="highlight">Smarter Surveillance</span></h1>
                        <p>
                            Welcome to Secure-AI, where advanced technology meets effortless security. We specialize in providing innovative solutions for seamless CCTV camera monitoring with AI-powered face tracking. Our cutting-edge technology allows you to effortlessly track a single individual across multiple cameras, ensuring comprehensive surveillance and enhanced security for your property or business.
                        </p>
                        <p>
                            At Secure-AI, we are committed to making surveillance smarter and more efficient. Our AI-driven face tracking system is designed to help you monitor key areas with ease and precision, giving you peace of mind knowing that every moment is being monitored accurately.
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
                        <img src="" alt="AI Security Camera System" />
                    </div>
                </div>
            </div>
            <div className="hero-wave">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320">
                    <path fill="#4B0082" fillOpacity="0.6" d="M0,192L48,176C96,160,192,128,288,133.3C384,139,480,181,576,208C672,235,768,245,864,224C960,203,1056,149,1152,138.7C1248,128,1344,160,1392,176L1440,192L1440,320L1392,320C1344,320,1248,320,1152,320C1056,320,960,320,864,320C768,320,672,320,576,320C480,320,384,320,288,320C192,320,96,320,48,320L0,320Z"></path>
                </svg>
            </div>
        </section>
    );
};

export default Hero;