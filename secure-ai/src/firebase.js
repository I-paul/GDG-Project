// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// Your Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyDy_k0ZBIHVtFuKegm8SmzIA7UMQ3ibADI",
  authDomain: "ai-72f78.firebaseapp.com",
  projectId: "ai-72f78",
  storageBucket: "ai-72f78.firebasestorage.app",
  messagingSenderId: "72552640869",
  appId: "1:72552640869:web:3e3b3d3cec88d552b7318c",
  measurementId: "G-M3KRTTPBNV"
};

// Initialize Firebase only once
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export { auth };