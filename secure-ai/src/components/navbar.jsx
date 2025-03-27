import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { useRef, useState, useEffect } from 'react';
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { User } from 'lucide-react';
import './styling/navbar.css';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { auth } from '../firebase';

const Navbar = () => {
    const navigate = useNavigate();
    const nav = useRef();
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [user, setUser] = useState(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    
    gsap.registerPlugin(useGSAP);
    
    useGSAP(() => {
        gsap.fromTo(nav.current, 
            {y: '-100%', ease: 'bounce'}, 
            {duration: 1.3, y: '0'});
    });

    useEffect(() => {
        // Monitor authentication state
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
        });
        
        return () => unsubscribe(); // Cleanup subscription
    }, []);

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

    const navigateToLogin = () => {
        navigate('/login');
        setMobileMenuOpen(false);
    };

    const toggleDropdown = () => {
        setDropdownOpen(!dropdownOpen);
    };

    const handleLogout = async () => {
        try {
            await signOut(auth);
            setDropdownOpen(false);
        } catch (error) {
            console.error("Error signing out: ", error);
        }
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
                {!user && (
                    <li className='mobile-auth'>
                        <div className='auth-button' onClick={navigateToLogin}>Login</div>
                    </li>
                )}
                {user && (
                    <li className='mobile-auth'>
                        <div className='profile-wrapper'>
                            <div className='user-profile' onClick={toggleDropdown}>
                                {user.photoURL ? (
                                    <img src={user.photoURL} alt="Profile" className="profile-image" />
                                ) : (
                                    <div className="profile-icon">
                                        <User size={24} color="white" />
                                    </div>
                                )}
                            </div>
                            {dropdownOpen && (
                                <div className='profile-dropdown mobile-dropdown'>
                                    <div className='dropdown-email'>{user.email}</div>
                                    <div className='dropdown-item' onClick={handleLogout}>Logout</div>
                                </div>
                            )}
                        </div>
                    </li>
                )}
            </ul>
            
            {!user ? (
                <div className='auth-button desktop-auth' onClick={navigateToLogin}>Login</div>
            ) : (
                <div className='profile-wrapper desktop-auth'>
                    <div className='user-profile' onClick={toggleDropdown}>
                        {user.photoURL ? (
                            <img src={user.photoURL} alt="Profile" className="profile-image" />
                        ) : (
                            <div className="profile-icon">
                                <User size={24} color="white" />
                            </div>
                        )}
                    </div>
                    {dropdownOpen && (
                        <div className='profile-dropdown'>
                            <div className='dropdown-email'>{user.email}</div>
                            <div className='dropdown-item' onClick={handleLogout}>Logout</div>
                        </div>
                    )}
                </div>
            )}
        </nav>
    );
};

export default Navbar;