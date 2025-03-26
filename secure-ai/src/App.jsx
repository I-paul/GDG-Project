import React, { Suspense, lazy } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import Navbar from "./components/navbar";
import "./App.css";
import './scripts/animations.js';

// Register GSAP plugin
gsap.registerPlugin(ScrollTrigger);

// Lazy load components
const Hero = lazy(() => import("./components/hero"));
const AddCam = lazy(() => import("./components/addcam"));
const Contact = lazy(() => import("./components/contact"));

// Loading fallback component
const PageLoader = () => (
  <div className="flex justify-center items-center h-screen">
    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-blue-500"></div>
  </div>
);

const App = () => {
  return (
    <div className="app-container">
      <Navbar />
      <Suspense fallback={<PageLoader />}>
        <Hero />
        <AddCam />
        <Contact />
      </Suspense>
    </div>
  );
};

export default App;