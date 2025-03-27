import React, { useRef, useState, useEffect } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { useNavigate } from 'react-router-dom';
import { User, LogOut, Menu, X } from 'lucide-react';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { auth } from '../firebase';
import './styling/navbar.css';

const Navbar = () => {
    const navigate = useNavigate();
    const navRef = useRef(null);
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [user, setUser] = useState(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    
    useGSAP(() => {
        // Entrance animation for navbar
        gsap.fromTo(navRef.current, 
            { y: '-100%', opacity: 0 }, 
            { 
                duration: 0.8, 
                y: '0', 
                opacity: 1, 
                ease: 'power3.out' 
            }
        );
    }, []);

    useEffect(() => {
        const handleScroll = () => {
            const scrollPosition = window.scrollY;
            const nav = navRef.current;
            
            if (scrollPosition > 50) {
                nav.classList.add('nav-scrolled');
            } else {
                nav.classList.remove('nav-scrolled');
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
        });
        
        return () => unsubscribe();
    }, []);

    const handleClick = (e, targetId) => {
        e.preventDefault();
        const target = document.querySelector(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
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
        <nav ref={navRef} className="navbar">
            <div className="navbar-container">
                <div className="navbar-logo">
                    <a href="#home" onClick={(e) => handleClick(e, '#home')}>
                        Secure-AI
                    </a>
                </div>

                {/* Mobile Menu Toggle */}
                <div 
                    className={`mobile-menu-toggle ${mobileMenuOpen ? 'active' : ''}`} 
                    onClick={toggleMobileMenu}
                >
                    {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                </div>

                {/* Navigation Links */}
                <ul className={`navbar-links ${mobileMenuOpen ? 'mobile-active' : ''}`}>
                    <li><a href="#home" onClick={(e) => handleClick(e, '#home')}>Home</a></li>
                    <li><a href="#monitor" onClick={(e) => handleClick(e, '#monitor')}>Monitor</a></li>
                    <li><a href="#contact" onClick={(e) => handleClick(e, '#contact')}>Contact</a></li>

                    {/* Authentication Section - Combined Mobile and Desktop */}
                    {!user ? (
                        <li className="auth-section">
                            <button onClick={navigateToLogin} className="auth-button">
                                Login
                            </button>
                        </li>
                    ) : (
                        <li className="auth-section">
                            <div className="user-profile" onClick={toggleDropdown}>
                                {user.photoURL ? (
                                    <img 
                                        src={user.photoURL} 
                                        alt="Profile" 
                                        className="profile-image" 
                                    />
                                ) : (
                                    <User size={24} color="white" />
                                )}
                            </div>
                            {dropdownOpen && (
                                <div className="profile-dropdown mobile-dropdown">
                                    <div className="dropdown-email">{user.email}</div>
                                    <div 
                                        className="dropdown-item" 
                                        onClick={handleLogout}
                                    >
                                        <LogOut size={16} /> Logout
                                    </div>
                                </div>
                            )}
                        </li>
                    )}
                </ul>
            </div>
        </nav>
    );
};

export default Navbar;