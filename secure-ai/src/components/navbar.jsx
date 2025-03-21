import {useGSAP} from '@gsap/react';
import gsap from 'gsap';
import { useRef, useState, useEffect } from 'react';
import React from 'react';
import './styling/navbar.css';

const Navbar = () => {
    const nav = useRef();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    
    gsap.registerPlugin(useGSAP,scroll);
    
    useGSAP(() => {
        gsap.fromTo(nav.current, 
            {y: '-100%', ease: 'bounce'}, 
            {duration: 1.3, y: '0'});
    });

    useEffect(() => {
        const handleScroll = () => {
            const scrollPosition = window.scrollY;
            if (scrollPosition > 100) {
                gsap.to(nav.current, {
                    backgroundColor: 'rgba(75, 0, 130, 0.95)',
                    backdropFilter: 'blur(10px)',
                    height: '70px',
                    duration: 0.3
                });
            } else {    
                gsap.to(nav.current, {
                    backgroundColor: 'transparent',
                    backdropFilter: 'blur(0px)',
                    height: '80px',
                    duration: 0.3
                });
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const handleClick = (e, targetId) => {
        e.preventDefault();
        const target = document.querySelector(targetId);
        if (target) {
            target.scrollIntoView({behavior: 'smooth'});
            setMobileMenuOpen(false);
        }
    };

    const toggleMobileMenu = () => {
        setMobileMenuOpen(!mobileMenuOpen);
    };

    return (
        <nav className='nav' ref={nav}>
            <div className="logo">
                <a href="#home" onClick={(e) => handleClick(e, '#home')}>Secure-AI</a>
            </div>
            
            <div className={`mobile-menu-icon ${mobileMenuOpen ? 'active' : ''}`} onClick={toggleMobileMenu}>
                <span></span>
                <span></span>
                <span></span>
            </div>
            
            <ul className={`nav-links ${mobileMenuOpen ? 'mobile-active' : ''}`}>
                <li className='link'><a href="#home" onClick={(e) => handleClick(e, '#home')}>Home</a></li>
                <li className='link'><a href="#monitor" onClick={(e) => handleClick(e, '#monitor')}>Monitor</a></li>
                <li className='link'><a href="#contact" onClick={(e) => handleClick(e, '#contact')}>Contact</a></li>
                <li className='mobile-auth'>
                    <div className='auth-button'>Login</div>
                </li>
            </ul>
            
            <div className='auth-button desktop-auth'>Login</div>
        </nav>
    );
};

export default Navbar;