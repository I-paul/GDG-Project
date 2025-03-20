import React, { useRef } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import Navbar from './components/navbar';
import Hero from './components/hero';
import AddCam from './components/addcam';
import './App.css';

gsap.registerPlugin(useGSAP,ScrollTrigger);

const App = () => {
  const appRef = useRef(null);
  const addCamRef = useRef(null);
  const heroRef = useRef(null);

  useGSAP(() => {
    gsap.fromTo(
      addCamRef.current,
      { x: "100%", opacity: 0 },
      {
        x: "0%",
        opacity: 1,
        duration: 1.5,
        ease: "power2.out",
        scrollTrigger: {
          trigger: heroRef.current,
          start: "top top",
          end: "bottom top",
          scrub: 1.5,
        },
      }
    );
  }, { scope: appRef });

  return (
    <div ref={appRef} className="app-container">
      <Navbar />
      <section id="home" ref={heroRef}>
        <Hero />
      </section>
      <section id="monitor" ref={addCamRef}>
        <AddCam />
      </section>
    </div>
  );
};

export default App;
