import React from 'react';
import { ChakraProvider, Box, Container } from '@chakra-ui/react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NavBar from './components/NavBar';
import CameraTracking from './components/CameraTracking';
import Dashboard from './components/Dashboard';
import theme from './theme';
import './App.css';

function App() {
  return (
    <ChakraProvider theme={theme}>
      <Router>
        <Box minH="100vh" bg="gray.50">
          <NavBar />
          <Container maxW="container.xl" py={6}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/camera-tracking" element={<CameraTracking />} />
              {/* Add other routes as needed */}
            </Routes>
          </Container>
        </Box>
      </Router>
    </ChakraProvider>
  );
}

export default App; 