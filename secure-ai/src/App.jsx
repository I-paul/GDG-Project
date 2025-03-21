import React, { useRef, useEffect, useState } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import Navbar from "./components/navbar";
import Hero from "./components/hero";
import AddCam from "./components/addcam";
import Contact from "./components/contact";
import "./App.css";

gsap.registerPlugin(ScrollTrigger);

const App = () => {
  const containerRef = useRef();
  const heroRef = useRef();
  const addCamRef = useRef();
  const contactRef = useRef();
  const videoRef = useRef();
  const [videoLoaded, setVideoLoaded] = useState(false);

  // Load video and prepare it for scroll control
  useEffect(() => {
    const video = videoRef.current;
    
    // Set video to be ready for scrubbing
    video.addEventListener('loadedmetadata', () => {
      video.pause();
      video.currentTime = 0;
      setVideoLoaded(true);
    });
    
    // Handle errors
    video.addEventListener('error', () => {
      console.error('Error loading video');
    });
    
    return () => {
      video.removeEventListener('loadedmetadata', () => {});
      video.removeEventListener('error', () => {});
    };
  }, []);

  useGSAP(() => {
    if (!videoLoaded) return;
    
    // Create a timeline for the entire page scroll
    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top top",
        end: "bottom bottom",
        scrub: 1.5, // Smooth scrubbing effect
        onUpdate: (self) => {
          const video = videoRef.current;
          if (video) {
            // Map scroll progress (0-1) to video duration
            const newTime = self.progress * video.duration;
            if (Math.abs(video.currentTime - newTime) > 0.01) {
              video.currentTime = newTime;
            }
          }
        }
      }
    });

    // Hero section animation
    gsap.fromTo(
      heroRef.current,
      { y: "0%", opacity: 1 }, 
      {
        y: "-100%",
        opacity: 0,
        ease: "power2.out",
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
    
  }, { scope: containerRef, dependencies: [videoLoaded] });

  return (
    <div className="app-container" ref={containerRef}>
      {/* Video Background */}
      <div className="video-background">
        <video 
          ref={videoRef}
          muted
          playsInline
          preload="auto"
          src="/path/to/your/background-animation.mp4"
        />
      </div>
      
      <Navbar />
      <Hero ref={heroRef} />
      <AddCam ref={addCamRef} />
      <Contact ref={contactRef} />
    </div>
  );
};

export default App;