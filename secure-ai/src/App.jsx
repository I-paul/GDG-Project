import { useState } from 'react'
import Navbar from './components/navbar';
import Hero from './components/hero';
import CamViewer from './components/camview';
import './App.css'

const App = () =>{
  return(
    <>
    <Navbar/>
    <Hero/>
    <CamViewer/>
    </>
  );
};

export default App
