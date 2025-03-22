import React, { useRef, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { gsap } from "gsap";
import { 
  GoogleAuthProvider, 
  signInWithEmailAndPassword, 
  createUserWithEmailAndPassword, 
  signInWithPopup, 
  sendPasswordResetEmail 
} from "firebase/auth";
import { auth } from "../firebase";
import '../components/styling/auth.css';

const LoginSignupPage = () => {
  const navigate = useNavigate();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [resetSent, setResetSent] = useState(false);
  
  const formRef = useRef(null);
  const formContentRef = useRef(null);
  
  // Google provider
  const googleProvider = new GoogleAuthProvider();
  
  useEffect(() => {
    // Animate form in
    if (formRef.current) {
      gsap.fromTo(
        formRef.current,
        { y: 50, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.5, ease: "power2.out" }
      );
    }
  }, []);
  
  useEffect(() => {
    // Animate form content when switching between login and signup
    if (formContentRef.current) {
      gsap.fromTo(
        formContentRef.current,
        { y: 20, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.4, ease: "power2.out" }
      );
    }
  }, [isLogin]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    
    try {
      if (isLogin) {
        // Login logic
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        // Sign up logic
        await createUserWithEmailAndPassword(auth, email, password);
      }
      
      // Successful login/signup animation
      if (formRef.current) {
        gsap.to(formRef.current, {
          scale: 1.05,
          duration: 0.2,
          ease: "power1.out",
          onComplete: () => {
            gsap.to(formRef.current, {
              scale: 1,
              duration: 0.3,
              ease: "elastic.out(1, 0.5)",
              onComplete: () => {
                // Navigate back to home page
                navigate('/');
              }
            });
          }
        });
      }
    } catch (err) {
      setError(err.message);
      
      // Error animation
      if (formRef.current) {
        gsap.to(formRef.current, {
          x: [-5, 5, -5, 5, 0],
          duration: 0.4,
          ease: "power1.inOut"
        });
      }
    } finally {
      setLoading(false);
    }
  };
  
  const handleGoogleSignIn = async () => {
    setError("");
    setLoading(true);
    try {
      await signInWithPopup(auth, googleProvider);
      navigate('/');
    } catch (err) {
      setError(err.message);
      
      if (formRef.current) {
        gsap.to(formRef.current, {
          x: [-5, 5, -5, 5, 0],
          duration: 0.4,
          ease: "power1.inOut"
        });
      }
    } finally {
      setLoading(false);
    }
  };
  
  const handleResetPassword = async () => {
    if (!email) {
      setError("Please enter your email address");
      return;
    }
    
    try {
      await sendPasswordResetEmail(auth, email);
      setResetSent(true);
    } catch (err) {
      setError(err.message);
    }
  };
  
  const handleGoBack = () => {
    navigate('/');
  };
  
  return (
    <div className="auth-page">
      <div className="auth-page-container">
        <div className="auth-form-wrapper" ref={formRef}>
          <button className="back-button" onClick={handleGoBack}>← Back to Home</button>
          
          <h2 className="auth-title">
            <span className="text-gradient">{isLogin ? "Welcome Back" : "Join Us"}</span>
          </h2>
          
          <div className="auth-tabs">
            <button 
              className={`auth-tab ${isLogin ? 'active' : ''}`}
              onClick={() => setIsLogin(true)}
              type="button"
            >
              Login
            </button>
            <button 
              className={`auth-tab ${!isLogin ? 'active' : ''}`}
              onClick={() => setIsLogin(false)}
              type="button"
            >
              Sign Up
            </button>
          </div>
          
          <form onSubmit={handleSubmit} ref={formContentRef}>
            {error && <div className="auth-error">{error}</div>}
            
            {resetSent && (
              <div className="auth-success">
                Password reset email sent! Check your inbox.
              </div>
            )}
            
            <div className="auth-input-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            
            <div className="auth-input-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            
            {isLogin && (
              <div className="auth-forgot-password">
                <button 
                  type="button" 
                  onClick={handleResetPassword}
                  className="auth-text-button"
                >
                  Forgot Password?
                </button>
              </div>
            )}
            
            <button 
              type="submit" 
              className="auth-submit-button"
              disabled={loading}
            >
              {loading ? (
                <span className="auth-loading-spinner"></span>
              ) : (
                isLogin ? "Login" : "Sign Up"
              )}
            </button>
            
            <div className="auth-divider">
              <span>or</span>
            </div>
            
            <button 
              type="button" 
              className="auth-google-button"
              onClick={handleGoogleSignIn}
              disabled={loading}
            >
              <svg viewBox="0 0 24 24" width="18" height="18">
                <path 
                  fill="currentColor" 
                  d="M12.545,10.239v3.821h5.445c-0.712,2.315-2.647,3.972-5.445,3.972c-3.332,0-6.033-2.701-6.033-6.032 s2.701-6.032,6.033-6.032c1.498,0,2.866,0.549,3.921,1.453l2.814-2.814C17.503,2.988,15.139,2,12.545,2 C7.021,2,2.543,6.477,2.543,12s4.478,10,10.002,10c8.396,0,10.249-7.85,9.426-11.748L12.545,10.239z"
                />
              </svg>
              Continue with Google
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};

export default LoginSignupPage;