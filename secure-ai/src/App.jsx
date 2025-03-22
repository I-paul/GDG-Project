// src/App.jsx
import React, { useRef, useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import Navbar from "./components/navbar";
import Hero from "./components/hero";
import AddCam from "./components/addcam";
import Contact from "./components/contact";
import "./App.css";

// Register GSAP plugin
gsap.registerPlugin(ScrollTrigger);

const App = () => {
  const containerRef = useRef(null);
  const heroRef = useRef(null);
  const addCamRef = useRef(null);
  const contactRef = useRef(null);

  useGSAP(() => {
    // Safely check if all refs are defined
    const allRefsExist = 
      containerRef.current && 
      heroRef.current && 
      addCamRef.current && 
      contactRef.current;
      
    if (!allRefsExist) {
      console.warn("Some refs are undefined. Skipping GSAP animations.");
      return;
    }
    
    // Basic section animations - simplified for stability
    
    // Hero section
    ScrollTrigger.create({
      trigger: heroRef.current,
      start: "top top",
      end: "bottom top",
      pin: true,
      pinSpacing: false
    });
    
    // AddCam section
    gsap.set(addCamRef.current, { y: "100vh" });
    
    ScrollTrigger.create({
      trigger: addCamRef.current,
      start: "top bottom",
      end: "top top",
      onEnter: () => {
        gsap.to(addCamRef.current, {
          y: "0vh",
          duration: 0.8,
          ease: "power2.out"
        });
      },
      onLeaveBack: () => {
        gsap.to(addCamRef.current, {
          y: "100vh",
          duration: 0.8,
          ease: "power2.in"
        });
      },
      pin: true,
      pinSpacing: false
    });
    
    // Contact section
    gsap.set(contactRef.current, { y: "100vh" });
    
    ScrollTrigger.create({
      trigger: contactRef.current,
      start: "top bottom",
      end: "top top",
      onEnter: () => {
        gsap.to(contactRef.current, {
          y: "0vh",
          duration: 0.8,
          ease: "power2.out"
        });
      },
      onLeaveBack: () => {
        gsap.to(contactRef.current, {
          y: "100vh",
          duration: 0.8,
          ease: "power2.in"
        });
      },
      pin: true,
      pinSpacing: false
    });
    
  }, { scope: containerRef, dependencies: [containerRef, heroRef, addCamRef, contactRef] });

  return (
    <div className="app-container" ref={containerRef}>
      <Navbar />
      <div>
        <Hero ref={heroRef} />
        <AddCam ref={addCamRef} />
        <Contact ref={contactRef} />
      </div>
    </div>
  );
};

export default App;