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
import { auth, db } from "../firebase";
import { doc, setDoc, getDoc, serverTimestamp } from "firebase/firestore";
import '../components/styling/auth.css';

// Validation helpers
const validateEmail = (email) => {
  const emailPattern = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailPattern.test(email);
};

const validatePassword = (password) => {
  // At least 6 characters, containing at least one number
  return password.length >= 6 && /\d/.test(password);
};

// Error message constants
const ERROR_MESSAGES = {
  INVALID_EMAIL: "Please enter a valid email address.",
  PASSWORD_REQUIREMENTS: "Password must be at least 6 characters and contain a number.",
  EMAIL_REQUIRED: "Please enter your email address.",
  GENERAL_ERROR: "Authentication error. Please try again.",
  USER_NOT_FOUND: "Account not found. Please check your email or sign up.",
  WRONG_PASSWORD: "Incorrect password. Please try again.",
  EMAIL_IN_USE: "This email is already in use. Try logging in instead.",
};

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
    // Ensure form element exists before animating
    const animateForm = () => {
      if (formRef.current) {
        gsap.fromTo(
          formRef.current,
          { y: 50, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.5, ease: "power2.out" }
        );
      }
    };
    
    // Small delay to ensure DOM is ready
    const timer = setTimeout(animateForm, 100);
    return () => clearTimeout(timer);
  }, []);
  
  useEffect(() => {
    // Ensure form content element exists before animating
    const animateFormContent = () => {
      if (formContentRef.current) {
        gsap.fromTo(
          formContentRef.current,
          { y: 20, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.4, ease: "power2.out" }
        );
      }
    };
    
    // Small delay to ensure DOM is ready
    const timer = setTimeout(animateFormContent, 100);
    return () => clearTimeout(timer);
  }, [isLogin]);

  // Function to create/update a user document in Firestore
  const updateUserDocument = async (userId, userEmail) => {
    try {
      // Reference to the user document
      const userDocRef = doc(db, "Users", userId);
      const userDoc = await getDoc(userDocRef);
      
      // Base user data
      const userData = {
        email: userEmail,
        last_login: serverTimestamp()
      };
      
      if (!userDoc.exists()) {
        // New user - create complete document
        await setDoc(userDocRef, {
          ...userData,
          connected_cameras: [],
          created_at: serverTimestamp(),
          display_name: userEmail.split('@')[0], // Default display name from email
          notifications_enabled: true,
          role: 'user',
          account_status: 'active'
        });
        console.log("New user document created");
      } else {
        // Existing user - just update login time
        await setDoc(userDocRef, userData, { merge: true });
        console.log("User document updated");
      }
      
      return true;
    } catch (error) {
      console.error("Error managing user document:", error);
      return false;
    }
  };
  
  const validateForm = () => {
    // Clear previous errors
    setError("");
    
    // Validate email
    if (!validateEmail(email)) {
      setError(ERROR_MESSAGES.INVALID_EMAIL);
      highlightField('email');
      return false;
    }
    
    // Validate password for sign up
    if (!isLogin && !validatePassword(password)) {
      setError(ERROR_MESSAGES.PASSWORD_REQUIREMENTS);
      highlightField('password');
      return false;
    }
    
    return true;
  };
  
  const highlightField = (fieldId) => {
    const field = document.getElementById(fieldId);
    if (field) {
      gsap.fromTo(
        field,
        { borderColor: "#ff6b6b" },
        { 
          borderColor: "rgba(255, 255, 255, 0.1)", 
          duration: 1,
          ease: "power2.inOut"
        }
      );
    }
  };
  
  const handleAuthError = (error) => {
    console.error("Auth error:", error.code);
    
    // Map Firebase error codes to user-friendly messages
    switch (error.code) {
      case 'auth/user-not-found':
        return ERROR_MESSAGES.USER_NOT_FOUND;
      case 'auth/wrong-password':
        return ERROR_MESSAGES.WRONG_PASSWORD;
      case 'auth/email-already-in-use':
        return ERROR_MESSAGES.EMAIL_IN_USE;
      default:
        return error.message || ERROR_MESSAGES.GENERAL_ERROR;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    setLoading(true);
    
    try {
      let userCredential;
      
      if (isLogin) {
        // Login logic
        userCredential = await signInWithEmailAndPassword(auth, email, password);
      } else {
        // Sign up logic
        userCredential = await createUserWithEmailAndPassword(auth, email, password);
      }
      
      // Update Firestore user document
      const success = await updateUserDocument(userCredential.user.uid, userCredential.user.email);
      
      if (success) {
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
      } else {
        setError("Error updating user profile. Please try again.");
      }
    } catch (err) {
      setError(handleAuthError(err));
      
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
      // Google sign in
      const result = await signInWithPopup(auth, googleProvider);
      const user = result.user;
      
      // Update Firestore user document
      const success = await updateUserDocument(user.uid, user.email);
      
      if (success) {
        // Successful login animation
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
                navigate('/');
              }
            });
          }
        });
      } else {
        setError("Error updating user profile. Please try again.");
      }
    } catch (err) {
      setError(handleAuthError(err));
      
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
      setError(ERROR_MESSAGES.EMAIL_REQUIRED);
      highlightField('email');
      return;
    }
    
    if (!validateEmail(email)) {
      setError(ERROR_MESSAGES.INVALID_EMAIL);
      highlightField('email');
      return;
    }
    
    setLoading(true);
    
    try {
      await sendPasswordResetEmail(auth, email);
      setResetSent(true);
      setError("");
      
      // Success animation
      const successMessage = document.querySelector('.auth-success');
      if (successMessage) {
        gsap.fromTo(
          successMessage,
          { opacity: 0, y: -10 },
          { opacity: 1, y: 0, duration: 0.4 }
        );
      }
    } catch (err) {
      setError(handleAuthError(err));
      
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
  
  const handleGoBack = () => {
    // Back button animation
    gsap.to('.auth-form-wrapper', {
      x: 20,
      opacity: 0,
      duration: 0.3,
      ease: "power2.in",
      onComplete: () => {
        navigate('/');
      }
    });
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
                onChange={(e) => {
                  setEmail(e.target.value);
                  setError(""); // Clear errors on change
                }}
                required
                autoComplete="email"
              />
            </div>
            
            <div className="auth-input-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => {
                  setPassword(e.target.value);
                  setError(""); // Clear errors on change
                }}
                required
                autoComplete={isLogin ? "current-password" : "new-password"}
              />
              {!isLogin && (
                <p className="password-hint">
                  Must be at least 6 characters with at least one number.
                </p>
              )}
            </div>
            
            {isLogin && (
              <div className="auth-forgot-password">
                <button 
                  type="button" 
                  onClick={handleResetPassword}
                  className="auth-text-button"
                  disabled={loading}
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