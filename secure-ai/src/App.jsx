import React, { useRef, useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { ScrollToPlugin } from "gsap/ScrollToPlugin";
import { useGSAP } from "@gsap/react";
import Navbar from "./components/navbar";
import Hero from "./components/hero";
import AddCam from "./components/addcam";
import Contact from "./components/contact";
import Auth from "./components/auth-UI";
import "./App.css";

// Register GSAP plugins
gsap.registerPlugin(ScrollTrigger, ScrollToPlugin);

const App = () => {
  const containerRef = useRef();
  const heroRef = useRef();
  const addCamRef = useRef();
  const contactRef = useRef();
  const authRef = useRef();

  useGSAP(() => {
    // Create a smooth scrolling effect
    const sections = [heroRef.current, addCamRef.current, authRef.current, contactRef.current];
    
    // Set initial states for sections
    gsap.set(sections.slice(1), { y: "100vh", opacity: 0 });
    
    // Create scroll-based timeline
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top top",
        end: "bottom bottom",
        scrub: 1,
        pin: true,
        anticipatePin: 1,
        snap: {
          snapTo: 1 / (sections.length - 1),
          duration: { min: 0.2, max: 0.5 },
          ease: "power1.inOut"
        }
      }
    });
    
    // Hero section animation with parallax elements
    const heroElements = heroRef.current.querySelectorAll('.hero-parallax');
    heroElements.forEach((el, i) => {
      gsap.fromTo(el, 
        { y: 0 },
        { 
          y: -50 * (i + 1), 
          scrollTrigger: {
            trigger: heroRef.current,
            start: "top top",
            end: "bottom top",
            scrub: true
          }
        }
      );
    });
    
    // Add subtle rotation to certain elements
    const rotateElements = heroRef.current.querySelectorAll('.hero-rotate');
    rotateElements.forEach((el) => {
      gsap.fromTo(el,
        { rotation: -5 },
        { 
          rotation: 5, 
          scrollTrigger: {
            trigger: heroRef.current,
            start: "top top",
            end: "bottom top",
            scrub: true
          }
        }
      );
    });
    
    // Hero to AddCam transition
    tl.to(heroRef.current, {
      y: "-100vh",
      opacity: 0,
      ease: "power2.inOut",
      duration: 0.5
    });
    
    // AddCam section animation
    tl.to(addCamRef.current, {
      y: "0vh",
      opacity: 1,
      ease: "power2.out",
      duration: 0.5
    }, "-=0.3");
    
    // AddCam to Auth transition
    tl.to(addCamRef.current, {
      y: "-100vh",
      opacity: 0,
      ease: "power2.inOut",
      duration: 0.5
    });
    
    // Auth section animation
    tl.to(authRef.current, {
      y: "0vh",
      opacity: 1,
      ease: "power2.out",
      duration: 0.5
    }, "-=0.3");
    
    // Auth to Contact transition
    tl.to(authRef.current, {
      y: "-100vh",
      opacity: 0,
      ease: "power2.inOut",
      duration: 0.5
    });
    
    // Contact section animation with staggered elements
    tl.to(contactRef.current, {
      y: "0vh",
      opacity: 1,
      ease: "power2.out",
      duration: 0.5
    }, "-=0.3");
    
    // Add staggered animation for contact elements
    const contactElements = contactRef.current.querySelectorAll('.contact-animate');
    gsap.set(contactElements, { y: 50, opacity: 0 });
    
    ScrollTrigger.create({
      trigger: contactRef.current,
      start: "top center",
      onEnter: () => {
        gsap.to(contactElements, {
          y: 0,
          opacity: 1,
          duration: 0.7,
          stagger: 0.1,
          ease: "back.out(1.7)"
        });
      },
      onLeaveBack: () => {
        gsap.to(contactElements, {
          y: 50,
          opacity: 0,
          duration: 0.5
        });
      }
    });
    
    // Add hover animations for interactive elements
    const buttons = document.querySelectorAll('button, .button, .interactive');
    buttons.forEach(button => {
      button.addEventListener('mouseenter', () => {
        gsap.to(button, {
          scale: 1.05,
          duration: 0.3,
          ease: "power1.out"
        });
      });
      
      button.addEventListener('mouseleave', () => {
        gsap.to(button, {
          scale: 1,
          duration: 0.3,
          ease: "power1.out"
        });
      });
    });
    
  }, { scope: containerRef });

  return (
    <div className="app-container" ref={containerRef}>
      <Navbar />
      <Hero ref={heroRef} />
      <AddCam ref={addCamRef} />
      <Auth ref={authRef} />
      <Contact ref={contactRef} />
    </div>
  );
};

export default App;