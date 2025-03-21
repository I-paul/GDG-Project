import React, { useRef, useEffect, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import Navbar from "./components/navbar";
import Hero from "./components/hero";
import AddCam from "./components/addcam";
import Contact from "./components/contact";
// import background from './assets/background.webm';
import "./App.css";

gsap.registerPlugin(ScrollTrigger);

const App = () => {
  const containerRef = useRef();
  const heroRef = useRef();
  const addCamRef = useRef();
  const contactRef = useRef();


  useGSAP(() => {

    // Hero section animation
    gsap.fromTo(
      heroRef.current,
      { y: "0%", opacity: 1 }, 
      {
        y: "-100%",
        opacity: 0,
        ease: "power2.out",
        zIndex: 1,
        scrollTrigger: {
          trigger: addCamRef.current,
          start: "top bottom",
          end: "top top",
          scrub: 1,
          pin: true,
          anticipatePin: 1,
        },
      }
    );

    // AddCam section animation
    gsap.fromTo(
      addCamRef.current,
      { y: "100vh", opacity: 0 },
      {
        y: "0vh",
        opacity: 1,
        ease: "power2.out",
        zIndex:3,
        scrollTrigger: {
          trigger: addCamRef.current,
          start: "top bottom",
          end: "top top",
          scrub: 1,
          pin: true,
          anticipatePin: 1,
        },
      }
    );
    
    // Contact section animation
    gsap.fromTo(
      contactRef.current,
      { y: "100vh", opacity: 0 },
      {
        y: "0vh",
        opacity: 1,
        ease: "power2.out",
        scrollTrigger: {
          trigger: contactRef.current,
          start: "top bottom",
          end: "top top",
          scrub: 1,
          pin: true,
          anticipatePin: 1,
        },
      }
    );
    
  }, { scope: containerRef });

  return (
    <div className="app-container" ref={containerRef}>
      <Navbar />
      <Hero ref={heroRef} />
      <AddCam ref={addCamRef} />
      <Contact ref={contactRef} />
    </div>
  );
};

export default App;