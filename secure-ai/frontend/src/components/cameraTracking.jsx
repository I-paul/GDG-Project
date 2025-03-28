import React, { useState, useEffect, useRef } from 'react';
import { Box, Button, Flex, Grid, Text, VStack, HStack, Slider, SliderTrack, SliderFilledTrack, SliderThumb, useToast, Input, Modal, ModalOverlay, ModalContent, ModalHeader, ModalFooter, ModalBody, ModalCloseButton, useDisclosure } from '@chakra-ui/react';
import { AddIcon, SettingsIcon, RepeatIcon, ViewIcon, CheckIcon } from '@chakra-ui/icons';
import './CameraTracking.css';

// Constants from the Python app
const DETECTION_CONFIDENCE = 0.35;
const MAX_FRAME_WIDTH = 400;
const DETECTION_INTERVAL = 100; // in ms
const MIN_DETECTION_INTERVAL = 80; // in ms
const FACE_DETECTION_INTERVAL = 200; // in ms
const MAX_FEATURE_CACHE = 10;
const SMOOTHING_FACTOR = 0.7;

const CameraTracking = () => {
  // State variables
  const [cv, setCv] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [cameras, setCameras] = useState([]);
  const [activeCameras, setActiveCameras] = useState({});
  const [cameraStreams, setCameraStreams] = useState({});
  const [trackingEnabled, setTrackingEnabled] = useState(true);
  const [crossCameraTracking, setCrossCameraTracking] = useState(true);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.75);
  const [selectedTrackId, setSelectedTrackId] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [lastSeenTimes, setLastSeenTimes] = useState({});
  const [fpsDisplay, setFpsDisplay] = useState(true);
  const [featureCache, setFeatureCache] = useState([]);
  const [smoothedBoxes, setSmoothedBoxes] = useState({});
  
  // Refs for canvas and video elements
  const canvasRefs = useRef({});
  const videoRefs = useRef({});
  const trackerRefs = useRef({});
  const detectionTimers = useRef({});
  const selectedPersonFeatures = useRef(null);
  const previewCanvasRef = useRef(null);
  
  // Modal for adding new cameras
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [newCameraUrl, setNewCameraUrl] = useState('');
  const [newCameraName, setNewCameraName] = useState('');
  const toast = useToast();

  // Load OpenCV.js
  useEffect(() => {
    if (window.cv) {
      setCv(window.cv);
      setIsLoaded(true);
      return;
    }

    // Define callback for when OpenCV.js is ready
    window.onOpenCvReady = () => {
      console.log('OpenCV.js is loaded');
      setCv(window.cv);
      setIsLoaded(true);
    };

    // Load OpenCV.js script
    const script = document.createElement('script');
    script.src = 'https://docs.opencv.org/4.8.0/opencv.js';
    script.async = true;
    script.type = 'text/javascript';
    document.body.appendChild(script);

    return () => {
      if (script.parentNode) {
        script.parentNode.removeChild(script);
      }
      window.onOpenCvReady = undefined;
    };
  }, []);

  // Initialize available cameras when OpenCV is loaded
  useEffect(() => {
    if (!isLoaded) return;
    
    // Get available local cameras
    const getLocalCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoCameras = devices.filter(device => device.kind === 'videoinput');
        
        const cameraList = videoCameras.map((camera, index) => ({
          id: `local_${index}`,
          name: camera.label || `Local Camera ${index}`,
          deviceId: camera.deviceId,
          type: 'local'
        }));
        
        setCameras(cameraList);
        
        // Initialize trackers for each camera
        cameraList.forEach(camera => {
          trackerRefs.current[camera.id] = new cv.TrackerKCF();
        });
        
        console.log('Local cameras detected:', cameraList.length);
      } catch (error) {
        console.error('Error detecting cameras:', error);
        toast({
          title: 'Camera Detection Error',
          description: error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };
    
    getLocalCameras();
  }, [isLoaded, toast]);

  // Function to start a camera stream
  const startCamera = async (camera) => {
    if (activeCameras[camera.id]) return;
    
    try {
      let stream;
      
      if (camera.type === 'local') {
        // Get local camera stream
        stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: camera.deviceId ? { exact: camera.deviceId } : undefined }
        });
      } else if (camera.type === 'ip') {
        // For IP cameras, we would need a server proxy
        // This is simplified for this example
        toast({
          title: 'IP Camera Note',
          description: 'IP cameras require server-side proxying which is not implemented in this frontend-only example',
          status: 'info',
          duration: 5000,
          isClosable: true,
        });
        return;
      }
      
      // Store the stream
      setCameraStreams(prev => ({ ...prev, [camera.id]: stream }));
      
      // Mark camera as active
      setActiveCameras(prev => ({ ...prev, [camera.id]: true }));
      
      // Setup the video element
      if (videoRefs.current[camera.id]) {
        videoRefs.current[camera.id].srcObject = stream;
        await videoRefs.current[camera.id].play();
        
        // Start processing frames
        startFrameProcessing(camera.id);
      }
      
      toast({
        title: 'Camera Started',
        description: `${camera.name} is now active`,
        status: 'success',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error(`Error starting camera ${camera.id}:`, error);
      toast({
        title: 'Camera Error',
        description: `Failed to start ${camera.name}: ${error.message}`,
        status: 'error',
        duration: 5000,
        isClosable: true,
      });
    }
  };

  // Function to stop a camera stream
  const stopCamera = (camera) => {
    if (!activeCameras[camera.id]) return;
    
    try {
      // Stop the stream
      if (cameraStreams[camera.id]) {
        cameraStreams[camera.id].getTracks().forEach(track => track.stop());
      }
      
      // Clear the timer
      if (detectionTimers.current[camera.id]) {
        clearTimeout(detectionTimers.current[camera.id]);
      }
      
      // Mark camera as inactive
      setActiveCameras(prev => {
        const newState = { ...prev };
        delete newState[camera.id];
        return newState;
      });
      
      // Remove the stream
      setCameraStreams(prev => {
        const newState = { ...prev };
        delete newState[camera.id];
        return newState;
      });
      
      toast({
        title: 'Camera Stopped',
        description: `${camera.name} has been stopped`,
        status: 'info',
        duration: 3000,
        isClosable: true,
      });
    } catch (error) {
      console.error(`Error stopping camera ${camera.id}:`, error);
    }
  };

  // Function to add a new IP camera
  const addIpCamera = () => {
    if (!newCameraUrl) {
      toast({
        title: 'Input Error',
        description: 'Please enter a camera URL',
        status: 'error',
        duration: 3000,
        isClosable: true,
      });
      return;
    }
    
    const newCamera = {
      id: `ip_${cameras.length}`,
      name: newCameraName || `IP Camera ${cameras.length}`,
      url: newCameraUrl,
      type: 'ip'
    };
    
    setCameras(prev => [...prev, newCamera]);
    
    // Initialize tracker for the new camera
    trackerRefs.current[newCamera.id] = new cv.TrackerKCF();
    
    toast({
      title: 'Camera Added',
      description: `${newCamera.name} has been added`,
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
    
    // Reset form
    setNewCameraUrl('');
    setNewCameraName('');
    onClose();
  };

  // Process frames from a camera
  const startFrameProcessing = (cameraId) => {
    if (!videoRefs.current[cameraId] || !canvasRefs.current[cameraId] || !cv) return;
    
    const video = videoRefs.current[cameraId];
    const canvas = canvasRefs.current[cameraId];
    const ctx = canvas.getContext('2d');
    
    // Ensure canvas dimensions match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    let lastDetectionTime = Date.now();
    let frameCount = 0;
    let fpsStartTime = Date.now();
    let fps = 0;
    
    // Setup detection throttler
    const throttler = {
      lastFaceDetection: 0,
      lastPersonDetection: 0,
      shouldRunFaceDetection: (currentTime, tracking = false) => {
        const interval = FACE_DETECTION_INTERVAL * (tracking ? 0.8 : 2.0);
        if (currentTime - throttler.lastFaceDetection >= interval) {
          throttler.lastFaceDetection = currentTime;
          return true;
        }
        return false;
      },
      shouldRunPersonDetection: (currentTime, faceCount = 0) => {
        if (faceCount > 0 && frameCount % 3 !== 0) return false;
        if (currentTime - throttler.lastPersonDetection >= DETECTION_INTERVAL) {
          throttler.lastPersonDetection = currentTime;
          return true;
        }
        return false;
      }
    };
    
    // Process a single frame
    const processFrame = () => {
      if (!activeCameras[cameraId]) return;
      
      try {
        // Draw the video frame on the canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Calculate FPS
        frameCount++;
        const currentTime = Date.now();
        if (currentTime - fpsStartTime >= 1000) {
          fps = frameCount / ((currentTime - fpsStartTime) / 1000);
          frameCount = 0;
          fpsStartTime = currentTime;
        }
        
        // Only run detection on intervals
        if (currentTime - lastDetectionTime >= DETECTION_INTERVAL) {
          lastDetectionTime = currentTime;
          
          // Create a Mat from the canvas
          const src = cv.imread(canvas);
          
          // Resize for faster processing
          const maxWidth = MAX_FRAME_WIDTH;
          let scaleFactor = 1.0;
          if (src.cols > maxWidth) {
            scaleFactor = maxWidth / src.cols;
            const dsize = new cv.Size(src.cols * scaleFactor, src.rows * scaleFactor);
            const resized = new cv.Mat();
            cv.resize(src, resized, dsize, 0, 0, cv.INTER_AREA);
            
            // Perform detection on resized image
            detectAndTrack(resized, canvas, ctx, cameraId, throttler, currentTime, scaleFactor);
            
            resized.delete();
          } else {
            // Perform detection on original image
            detectAndTrack(src, canvas, ctx, cameraId, throttler, currentTime, 1.0);
          }
          
          // Clean up
          src.delete();
        }
        
        // Display FPS if enabled
        if (fpsDisplay) {
          ctx.font = '16px Arial';
          ctx.fillStyle = 'lime';
          ctx.fillText(`FPS: ${fps.toFixed(1)}`, 10, 20);
        }
        
        // Update tracking status
        ctx.font = '16px Arial';
        ctx.fillStyle = trackingEnabled ? 'lime' : 'red';
        ctx.fillText(trackingEnabled ? 'TRACKING ON' : 'TRACKING OFF', 10, canvas.height - 10);
        
        // Display last seen information if tracking is enabled
        if (trackingEnabled && selectedPersonFeatures.current) {
          let seenText = '';
          let textColor = 'red';
          
          if (lastSeenTimes[cameraId]) {
            const timeSinceSeen = (currentTime - lastSeenTimes[cameraId]) / 1000;
            if (timeSinceSeen < 5.0) {
              seenText = 'PERSON PRESENT';
              textColor = 'lime';
            } else {
              seenText = `Last seen: ${timeSinceSeen.toFixed(1)}s ago`;
              textColor = 'orange';
            }
          } else {
            seenText = 'Not seen yet';
          }
          
          ctx.fillStyle = textColor;
          ctx.fillText(seenText, canvas.width - 200, canvas.height - 10);
        }
        
        // Schedule next frame
        detectionTimers.current[cameraId] = requestAnimationFrame(processFrame);
      } catch (error) {
        console.error(`Error processing frame for camera ${cameraId}:`, error);
        // Try to recover
        detectionTimers.current[cameraId] = requestAnimationFrame(processFrame);
      }
    };
    
    // Start processing
    processFrame();
  };

  // Detect people and faces in a frame and track them
  const detectAndTrack = (frame, canvas, ctx, cameraId, throttler, currentTime, scaleFactor) => {
    if (!cv) return;
    
    try {
      let detections = [];
      let facesFound = 0;
      
      // Run face detection if needed
      if (throttler.shouldRunFaceDetection(currentTime, selectedPersonFeatures.current !== null)) {
        // We would use OpenCV's face detector here
        // Since OpenCV.js doesn't include face detection, we'd need a face-api.js integration
        // Simulating face detection for this example:
        
        // Convert frame to grayscale for face detection
        const gray = new cv.Mat();
        cv.cvtColor(frame, gray, cv.COLOR_RGBA2GRAY);
        
        // Use a classifier from OpenCV.js (would need to be loaded)
        // In a real implementation, we would use face-api.js here
        // For now, we'll skip actual detection
        
        gray.delete();
        
        // Simulate some face detections for demo purposes
        // In a real app, these would come from actual detection
        if (Math.random() > 0.7) {  // Randomly detect a face 30% of the time
          const fakeX = Math.floor(Math.random() * frame.cols * 0.6) + frame.cols * 0.2;
          const fakeY = Math.floor(Math.random() * frame.rows * 0.6) + frame.rows * 0.1;
          const fakeW = Math.floor(Math.random() * 60) + 60;
          const fakeH = Math.floor(Math.random() * 30) + 90;
          
          // Scale back to original size
          const x = Math.floor(fakeX / scaleFactor);
          const y = Math.floor(fakeY / scaleFactor);
          const w = Math.floor(fakeW / scaleFactor);
          const h = Math.floor(fakeH / scaleFactor);
          
          // Add to detections list
          detections.push({
            bbox: [x, y, w, h],
            confidence: 0.95,
            classId: 0,
            features: generateRandomFeatureVector(128)  // Simulate embedding features
          });
          
          facesFound = 1;
        }
      }
      
      // Run person detection if needed
      if (throttler.shouldRunPersonDetection(currentTime, facesFound)) {
        // We would use YOLO or another detector here
        // For this example, we'll simulate person detection
        
        // Simulate person detection with random boxes
        if (Math.random() > 0.5) {  // 50% chance to detect a person
          const numDetections = Math.floor(Math.random() * 2) + 1;  // 1-2 people
          
          for (let i = 0; i < numDetections; i++) {
            const x = Math.floor(Math.random() * frame.cols * 0.6) + frame.cols * 0.1;
            const y = Math.floor(Math.random() * frame.rows * 0.4) + frame.rows * 0.3;
            const w = Math.floor(Math.random() * 80) + 100;
            const h = Math.floor(Math.random() * 60) + 200;
            
            // Scale back to original size
            const scaledX = Math.floor(x / scaleFactor);
            const scaledY = Math.floor(y / scaleFactor);
            const scaledW = Math.floor(w / scaleFactor);
            const scaledH = Math.floor(h / scaleFactor);
            
            // Add to detections
            detections.push({
              bbox: [scaledX, scaledY, scaledW, scaledH],
              confidence: Math.random() * 0.3 + 0.65,  // 0.65-0.95
              classId: 0,
              features: generateRandomFeatureVector(128)  // Simulate embedding features
            });
          }
        }
      }
      
      // Process detections - draw boxes and match with tracked person
      if (detections.length > 0) {
        let bestMatch = null;
        let bestSimilarity = 0;
        
        // Draw all detections and find best match if tracking
        for (const [index, detection] of detections.entries()) {
          const [x, y, w, h] = detection.bbox;
          const trackId = index + 1;  // Simple ID assignment
          
          // Smooth box if tracking
          let displayBox = [x, y, w, h];
          if (trackingEnabled) {
            displayBox = stabilizeBox(trackId, [x, y, w, h], smoothedBoxes, setSmoothedBoxes);
          }
          
          // Check for match with selected person if tracking
          let isMatch = false;
          let similarity = 0;
          
          if (trackingEnabled && selectedPersonFeatures.current !== null && detection.features) {
            similarity = cosineSimilarity(selectedPersonFeatures.current, detection.features);
            
            if (similarity > similarityThreshold) {
              isMatch = true;
              
              // Keep track of best match
              if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = { ...detection, trackId, similarity };
              }
            }
          }
          
          // Draw box
          const color = isMatch ? `rgb(0, ${Math.min(255, 128 + 127 * similarity)}, 0)` : getRandomColor(trackId);
          const thickness = isMatch ? 2 : 1;
          
          // Draw bounding box
          ctx.strokeStyle = color;
          ctx.lineWidth = thickness;
          ctx.strokeRect(displayBox[0], displayBox[1], displayBox[2], displayBox[3]);
          
          // Draw label
          const label = `ID: ${trackId}${isMatch ? ` (${similarity.toFixed(2)})` : ''}`;
          ctx.font = '12px Arial';
          ctx.fillStyle = color;
          ctx.fillRect(displayBox[0], displayBox[1] - 20, ctx.measureText(label).width + 10, 20);
          ctx.fillStyle = 'white';
          ctx.fillText(label, displayBox[0] + 5, displayBox[1] - 5);
          
          // Draw target corners for tracked person
          if (isMatch) {
            const cornerLength = Math.min(20, Math.min(displayBox[2], displayBox[3]) / 3);
            
            ctx.strokeStyle = 'yellow';
            ctx.lineWidth = 2;
            
            // Top-left corner
            ctx.beginPath();
            ctx.moveTo(displayBox[0], displayBox[1]);
            ctx.lineTo(displayBox[0] + cornerLength, displayBox[1]);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(displayBox[0], displayBox[1]);
            ctx.lineTo(displayBox[0], displayBox[1] + cornerLength);
            ctx.stroke();
            
            // Top-right corner
            ctx.beginPath();
            ctx.moveTo(displayBox[0] + displayBox[2], displayBox[1]);
            ctx.lineTo(displayBox[0] + displayBox[2] - cornerLength, displayBox[1]);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(displayBox[0] + displayBox[2], displayBox[1]);
            ctx.lineTo(displayBox[0] + displayBox[2], displayBox[1] + cornerLength);
            ctx.stroke();
            
            // Bottom-left corner
            ctx.beginPath();
            ctx.moveTo(displayBox[0], displayBox[1] + displayBox[3]);
            ctx.lineTo(displayBox[0] + cornerLength, displayBox[1] + displayBox[3]);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(displayBox[0], displayBox[1] + displayBox[3]);
            ctx.lineTo(displayBox[0], displayBox[1] + displayBox[3] - cornerLength);
            ctx.stroke();
            
            // Bottom-right corner
            ctx.beginPath();
            ctx.moveTo(displayBox[0] + displayBox[2], displayBox[1] + displayBox[3]);
            ctx.lineTo(displayBox[0] + displayBox[2] - cornerLength, displayBox[1] + displayBox[3]);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(displayBox[0] + displayBox[2], displayBox[1] + displayBox[3]);
            ctx.lineTo(displayBox[0] + displayBox[2], displayBox[1] + displayBox[3] - cornerLength);
            ctx.stroke();
          }
        }
        
        // Update last seen time if best match found
        if (bestMatch) {
          setLastSeenTimes(prev => ({ ...prev, [cameraId]: currentTime }));
          
          // Update feature vector with adaptive weighting
          if (selectedPersonFeatures.current !== null) {
            // Add to feature cache
            setFeatureCache(prev => {
              const newCache = [...prev, bestMatch.features];
              if (newCache.length > MAX_FEATURE_CACHE) newCache.shift();
              return newCache;
            });
            
            // Update with gradually increasing weight based on confidence
            if (bestSimilarity > 0.9) {
              // Very confident match
              selectedPersonFeatures.current = weightedAverage(
                selectedPersonFeatures.current, 
                bestMatch.features,
                0.85, 0.15
              );
            } else if (bestSimilarity > 0.8) {
              // Good match
              selectedPersonFeatures.current = weightedAverage(
                selectedPersonFeatures.current, 
                bestMatch.features,
                0.9, 0.1
              );
            } else {
              // Weak match
              selectedPersonFeatures.current = weightedAverage(
                selectedPersonFeatures.current, 
                bestMatch.features,
                0.98, 0.02
              );
            }
            
            // Normalize the feature vector
            selectedPersonFeatures.current = normalizeVector(selectedPersonFeatures.current);
            
            // Update preview image
            const [x, y, w, h] = bestMatch.bbox;
            const previewCanvas = document.createElement('canvas');
            previewCanvas.width = w;
            previewCanvas.height = h;
            
            const previewCtx = previewCanvas.getContext('2d');
            previewCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h);
            
            setPreviewImage(previewCanvas.toDataURL());
          }
        }
      }
    } catch (error) {
      console.error('Error in detectAndTrack:', error);
    }
  };

  // Stabilize bounding boxes to reduce flickering
  const stabilizeBox = (trackId, newBox, boxes, setBoxes) => {
    const [newX, newY, newW, newH] = newBox;
    
    if (!boxes[trackId]) {
      setBoxes(prev => ({ ...prev, [trackId]: newBox }));
      return newBox;
    }
    
    const [x, y, w, h] = boxes[trackId];
    
    // Apply exponential moving average
    const alpha = SMOOTHING_FACTOR;
    const smoothX = Math.round(alpha * newX + (1 - alpha) * x);
    const smoothY = Math.round(alpha * newY + (1 - alpha) * y);
    const smoothW = Math.round(alpha * newW + (1 - alpha) * w);
    const smoothH = Math.round(alpha * newH + (1 - alpha) * h);
    
    const smoothedBox = [smoothX, smoothY, smoothW, smoothH];
    setBoxes(prev => ({ ...prev, [trackId]: smoothedBox }));
    
    return smoothedBox;
  };

  // Handle canvas click to select a person for tracking
  const handleCanvasClick = (e, cameraId) => {
    if (!canvasRefs.current[cameraId]) return;
    
    const canvas = canvasRefs.current[cameraId];
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Simulate selecting a person - in a real app, we would find the detection at this position
    const selectedId = Math.floor(Math.random() * 10) + 1;  // Random track ID
    const selectedFeatures = generateRandomFeatureVector(128);  // Random feature vector
    
    // Set the selected person
    setSelectedTrackId(selectedId);
    selectedPersonFeatures.current = selectedFeatures;
    setTrackingEnabled(true);
    
    // Clear feature cache
    setFeatureCache([]);
    setSmoothedBoxes({});
    
    // Reset last seen times
    setLastSeenTimes({ [cameraId]: Date.now() });
    
    // Create a preview image
    setPreviewImage(canvas.toDataURL());
    
    toast({
      title: 'Person Selected',
      description: `Now tracking Person ID: ${selectedId}`,
      status: 'success',
      duration: 3000,
      isClosable: true,
    });
  };

  // Helper function to generate a random feature vector
  const generateRandomFeatureVector = (size) => {
    const vector = new Array(size).fill(0).map(() => Math.random() * 2 - 1);
    return normalizeVector(vector);
  };

  // Helper function to normalize a vector
  const normalizeVector = (vector) => {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return vector.map(val => val / (magnitude || 1));
  };

  // Helper function to compute cosine similarity
  const cosineSimilarity = (vectorA, vectorB) => {
    if (vectorA.length !== vectorB.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < vectorA.length; i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += vectorA[i] * vectorA[i];
      normB += vectorB[i] * vectorB[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
  };

  // Helper function for weighted average of vectors
  const weightedAverage = (vectorA, vectorB, weightA, weightB) => {
    return vectorA.map((val, i) => weightA * val + weightB * vectorB[i]);
  };

  // Helper function to get a consistent color for a track ID
  const getRandomColor = (id) => {
    const colorSeed = Math.abs(id * 1000) % 360;
    return `hsl(${colorSeed}, 70%, 50%)`;
  };

  // Clean up resources when component unmounts
  useEffect(() => {
    return () => {
      // Stop all camera streams
      Object.entries(cameraStreams).forEach(([id, stream]) => {
        stream.getTracks().forEach(track => track.stop());
      });
      
      // Clear all timers
      Object.values(detectionTimers.current).forEach(timer => {
        cancelAnimationFrame(timer);
      });
    };
  }, [cameraStreams]);

  // Render loading state if OpenCV.js isn't ready
  if (!isLoaded) {
    return (
      <Box p={5} textAlign="center">
        <Text fontSize="xl">Loading OpenCV.js...</Text>
      </Box>
    );
  }

  return (
    <Box p={4}>
      <Flex justify="space-between" align="center" mb={4}>
        <Text fontSize="2xl" fontWeight="bold">Multi-Camera Tracking System</Text>
        <HStack>
          <Button 
            leftIcon={<AddIcon />} 
            colorScheme="blue" 
            onClick={onOpen}
          >
            Add Camera
          </Button>
          <Button
            leftIcon={<SettingsIcon />}
            colorScheme={trackingEnabled ? "green" : "red"}
            onClick={() => setTrackingEnabled(!trackingEnabled)}
          >
            {trackingEnabled ? "Tracking ON" : "Tracking OFF"}
          </Button>
          <Button
            leftIcon={<RepeatIcon />}
            colorScheme="red"
            onClick={() => {
              setSelectedTrackId(null);
              selectedPersonFeatures.current = null;
              setPreviewImage(null);
              setLastSeenTimes({});
              setFeatureCache([]);
              setSmoothedBoxes({});
            }}
          >
            Reset Tracking
          </Button>
        </HStack>
      </Flex>

      <Grid templateColumns="repeat(auto-fill, minmax(320px, 1fr))" gap={6}>
        {cameras.map((camera) => (
          <Box
            key={camera.id}
            borderWidth="1px"
            borderRadius="lg"
            overflow="hidden"
            boxShadow="md"
          >
            <Box p={3} bg="gray.100">
              <Flex justify="space-between" align="center">
                <Text fontWeight="bold">{camera.name}</Text>
                <Button
                  colorScheme={activeCameras[camera.id] ? "red" : "green"}
                  size="sm"
                  onClick={() => activeCameras[camera.id] ? stopCamera(camera) : startCamera(camera)}
                >
                  {activeCameras[camera.id] ? "Stop" : "Start"}
                </Button>
              </Flex>
            </Box>
            
            <Box position="relative" height="240px" bg="black">
              <video
                ref={el => videoRefs.current[camera.id] = el}
                style={{ display: 'none' }}
                muted
                playsInline
              />
              <canvas
                ref={el => canvasRefs.current[camera.id] = el}
                style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                onClick={(e) => handleCanvasClick(e, camera.id)}
              />
              
              {!activeCameras[camera.id] && (
                <Flex
                  position="absolute"
                  top="0"
                  left="0"
                  width="100%"
                  height="100%"
                  justify="center"
                  align="center"
                  bg="rgba(0, 0, 0, 0.7)"
                >
                  <Button
                    colorScheme="green"
                    leftIcon={<ViewIcon />}
                    onClick={() => startCamera(camera)}
                  >
                    View Camera
                  </Button>
                </Flex>
              )}
            </Box>
            
            <Box p={3}>
              <Text fontSize="sm" color="gray.500">
                {camera.type === 'local' ? `Local Camera` : `IP Camera: ${camera.url}`}
              </Text>
            </Box>
          </Box>
        ))}
      </Grid>

      {previewImage && (
        <Box mt={6} p={4} borderWidth="1px" borderRadius="lg" boxShadow="md">
          <Text fontSize="xl" fontWeight="bold" mb={2}>Tracked Person</Text>
          <Flex direction={{ base: "column", md: "row" }} align="start">
            <Box width={{ base: "100%", md: "200px" }} mb={{ base: 4, md: 0 }} mr={{ md: 6 }}>
              <img 
                src={previewImage} 
                alt="Tracked person" 
                style={{ width: '100%', height: 'auto', border: '2px solid green' }}
              />
              <Text mt={2} fontWeight="bold">
                Track ID: {selectedTrackId || 'None'}
              </Text>
            </Box>
            
            <Box flex="1">
              <Text mb={2}>Similarity Threshold: {similarityThreshold.toFixed(2)}</Text>
              <Slider
                min={0.1}
                max={1.0}
                step={0.01}
                value={similarityThreshold}
                onChange={setSimilarityThreshold}
                mb={4}
              >
                <SliderTrack>
                  <SliderFilledTrack />
                </SliderTrack>
                <SliderThumb />
              </Slider>
              
              <Flex justify="space-between" mb={4}>
                <Button
                  colorScheme={crossCameraTracking ? "green" : "red"}
                  onClick={() => setCrossCameraTracking(!crossCameraTracking)}
                  leftIcon={<CheckIcon />}
                  size="sm"
                >
                  {crossCameraTracking ? "Cross-Camera ON" : "Cross-Camera OFF"}
                </Button>
                
                <Button
                  colorScheme={fpsDisplay ? "green" : "gray"}
                  onClick={() => setFpsDisplay(!fpsDisplay)}
                  size="sm"
                >
                  {fpsDisplay ? "FPS Display ON" : "FPS Display OFF"}
                </Button>
              </Flex>
            </Box>
          </Flex>
        </Box>
      )}

      {/* Modal for adding IP camera */}
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Add IP Camera</ModalHeader>
          <ModalCloseButton />
          <ModalBody>
            <VStack spacing={4}>
              <Input
                placeholder="Camera Name (optional)"
                value={newCameraName}
                onChange={(e) => setNewCameraName(e.target.value)}
              />
              <Input
                placeholder="RTSP URL or IP address"
                value={newCameraUrl}
                onChange={(e) => setNewCameraUrl(e.target.value)}
              />
              <Text fontSize="sm" color="gray.500">
                Example: rtsp://username:password@192.168.1.64:554/Streaming/Channels/1
              </Text>
            </VStack>
          </ModalBody>
          <ModalFooter>
            <Button colorScheme="blue" mr={3} onClick={addIpCamera}>
              Add Camera
            </Button>
            <Button variant="ghost" onClick={onClose}>Cancel</Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </Box>
  );
};

export default CameraTracking;