import {useGSAP} from '@gsap/react';
import gsap from 'gsap';
import { useRef } from 'react';
import React from 'react';
import './styling/navbar.css';

const Navbar = () =>{
    const nav = useRef();
    gsap.registerPlugin(useGSAP);
    useGSAP(()=>{
        gsap.fromTo(nav.current,{duration:1.3,y:'-100%',ease:'bounce'},{duration:1.3,y:'1%',ease:'bounce'});
    }
    );

    const anchors = document.querySelectorAll('a[href^="#"]');

    anchors.forEach(anchor => {
      anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = document.querySelector(this.getAttribute('href'));
        targetId.scrollIntoView({behavior: 'smooth'});
      });
    });
    return (
        <nav className='nav' ref={nav}>
            <div className="logo">
                <a href="#">Secure-AI</a>
            </div>
            <ul className='nav-links'>
                <li className='link'><a href="#home">Home</a></li>
                <li className='link'><a href="#monitor">Monitor</a></li>
                <li className='link'><a href="#contact">Contact</a></li>
            </ul>
            <div className='auth-button' onClick={''}>Login</div>
        </nav>
    );
};

export default Navbar;