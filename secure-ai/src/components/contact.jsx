import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { Mail, Facebook, Twitter, Linkedin, Instagram, Github, Dribbble} from 'lucide-react';
import './styling/contact.css';

const Contact = () => {
    const contactRef = useRef(null);
    const infoRef = useRef(null);
    const creatorsRef = useRef(null);

    useGSAP(() => {
        // Simple fade-in animations
        gsap.from(infoRef.current, {
            opacity: 0,
            duration: 0.8,
            ease: "power3.out",
            scrollTrigger:{trigger:contactRef.current,
                start: 'top 35%', 
                end: '+=500', 
                scrub: 1, toggleActions : 'restart pause reverse reverse'},
        });
        
        gsap.from(creatorsRef.current, {
            opacity: 0,
            duration: 0.8,
            ease: "power3.out",
            delay: 0.2,
            scrollTrigger:{trigger:contactRef.current,
                start: 'top 35%', 
                end: '+=500', 
                scrub: 1, toggleActions : 'restart pause reverse reverse'},
        });
    }, { scope: contactRef });

    return (
        <section id="contact" ref={contactRef}>
            <div className="contact-container">
                <h2 className="section-title">Get in <span className="highlight">Touch</span></h2>
                
                {/* <div className="contact-content">
                    <div className="contact-info" ref={infoRef}>
                        <h3>Contact Information</h3>
                        
                        <div className="info-item">
                            <div className="info-icon">
                                <Mail />
                            </div>
                            <div className="info-content">
                                <h4>Email</h4>
                                <p>secure-ai@gmail.com</p>
                            </div>
                        </div>
                        
                        <div className="social-links">
                            <a href="#" className="social-link"><Facebook /></a>
                            <a href="#" className="social-link"><Twitter /></a>
                            <a href="#" className="social-link"><Linkedin /></a>
                            <a href="#" className="social-link"><Instagram /></a>
                        </div>
                    </div>
                </div> */}
                
                <div className="creators-section" ref={creatorsRef}>
                    <h3>Meet Our Creators</h3>
                    <div className="creators-container">
                        <div className="creator">
                            <h4>Israel Paul</h4>
                            <p>Web Developer</p>
                            <div className="creator-socials">
                                <a href="#" className="social-link"><Github size={20} /></a>
                                <a href="#" className="social-link"><Linkedin size={20} /></a>
                                <a href="#" className="social-link"><Mail size={20} /></a>
                            </div>
                        </div>
                        
                        <div className="creator">
                            <h4>Ajay S Vasan</h4>
                            <p>AI Engineer</p>
                            <div className="creator-socials">
                                {/* <a href="#" className="social-link"><Dribbble size={20} /></a> */}
                                <a href="#" className="social-link"><Github size={20} /></a>
                                <a href="#" className="social-link"><Linkedin size={20} /></a>
                                <a href="#" className="social-link"><Mail size={20} /></a>
                            </div>
                        </div>
                        
                        <div className="creator">
                            <h4>Diviyan Frank Jeyasingh</h4>
                            <p>Web Developer</p>
                            <div className="creator-socials">
                                <a href="#" className="social-link"><Github size={20} /></a>
                                <a href="#" className="social-link"><Linkedin size={20} /></a>
                                <a href="#" className="social-link"><Mail size={20} /></a>
                            </div>
                        </div>
                        {/* <div className="creator">
                            <h4>Diviyan Frank Jeyasingh</h4>
                            <p>Web Developer</p>
                            <div className="creator-socials">
                                <a href="#" className="social-link"><Github size={20} /></a>
                                <a href="#" className="social-link"><Linkedin size={20} /></a>
                            </div>
                        </div> */}
                    </div>
                </div>
                
                <div className="copyright">
                    <p>&copy; 2025 Secure-AI. All rights reserved.</p>
                    <div className="footer-links">
                        <a href="#">Privacy Policy</a>
                        <a href="#">Terms of Service</a>
                        <a href="#">Cookie Policy</a>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default Contact;