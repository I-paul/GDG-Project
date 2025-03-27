import React, { useRef, useEffect } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { Mail, Facebook, Twitter, Linkedin, Instagram, Github } from 'lucide-react';
import './styling/contact.css';

// Register ScrollTrigger plugin
gsap.registerPlugin(ScrollTrigger);

const Contact = () => {
    const contactRef = useRef(null);
    const creatorsRef = useRef(null);

    useEffect(() => {
        if (!contactRef.current || !creatorsRef.current) {
            console.warn("One or more animation targets are missing.");
            return;
        }

        // Animation for the contact section
        gsap.fromTo(
            contactRef.current,
            { opacity: 0, y: 100, scale: 0.9 },
            {
                opacity: 1,
                y: 0,
                scale: 1,
                duration: 1.2,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: contactRef.current,
                    start: 'top 70%',
                    toggleActions: 'play none none reverse',
                },
            }
        );

        // Animation for the creators section
        const creatorElements = creatorsRef.current.querySelectorAll('.creator');
        gsap.fromTo(
            creatorElements,
            { opacity: 0, x: -50, rotate: -10 },
            {
                opacity: 1,
                x: 0,
                rotate: 0,
                stagger: 0.2,
                duration: 1,
                ease: 'back.out(1.7)',
                scrollTrigger: {
                    trigger: creatorsRef.current,
                    start: 'top 70%',
                    toggleActions: 'play none none reverse',
                },
            }
        );
    }, []);

    return (
        <section id="contact" ref={contactRef}>
            <div className="contact-container">
                <h2 className="section-title">Get in <span className="highlight">Touch</span></h2>

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