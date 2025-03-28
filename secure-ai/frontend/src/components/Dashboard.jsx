import React from 'react';
import { Box, Heading, SimpleGrid, Stat, StatLabel, StatNumber, StatHelpText, StatArrow, Button, Text, VStack, HStack, Icon } from '@chakra-ui/react';
import { FiCamera, FiUsers, FiClock, FiSettings } from 'react-icons/fi';
import { Link as RouterLink } from 'react-router-dom';

const Dashboard = () => {
  return (
    <Box>
      <Heading mb={6}>Dashboard</Heading>
      
      <SimpleGrid columns={{ base: 1, md: 4 }} spacing={6} mb={8}>
        <Stat p={4} shadow="md" border="1px" borderColor="gray.200" borderRadius="md">
          <StatLabel>Active Cameras</StatLabel>
          <StatNumber>3</StatNumber>
          <StatHelpText>
            <StatArrow type="increase" />
            23% increase
          </StatHelpText>
        </Stat>
        
        <Stat p={4} shadow="md" border="1px" borderColor="gray.200" borderRadius="md">
          <StatLabel>People Tracked</StatLabel>
          <StatNumber>42</StatNumber>
          <StatHelpText>
            <StatArrow type="increase" />
            12% increase
          </StatHelpText>
        </Stat>
        
        <Stat p={4} shadow="md" border="1px" borderColor="gray.200" borderRadius="md">
          <StatLabel>Detection Rate</StatLabel>
          <StatNumber>95%</StatNumber>
          <StatHelpText>
            <StatArrow type="increase" />
            5% increase
          </StatHelpText>
        </Stat>
        
        <Stat p={4} shadow="md" border="1px" borderColor="gray.200" borderRadius="md">
          <StatLabel>System Uptime</StatLabel>
          <StatNumber>3d 12h</StatNumber>
          <StatHelpText>
            <StatArrow type="decrease" />
            2 restarts
          </StatHelpText>
        </Stat>
      </SimpleGrid>
      
      <SimpleGrid columns={{ base: 1, md: 3 }} spacing={6}>
        <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
          <VStack align="start" spacing={4}>
            <HStack>
              <Icon as={FiCamera} boxSize={6} color="blue.500" />
              <Heading size="md">Camera Management</Heading>
            </HStack>
            <Text>Configure and manage IP and local cameras for tracking.</Text>
            <Button as={RouterLink} to="/camera-tracking" colorScheme="blue">
              Manage Cameras
            </Button>
          </VStack>
        </Box>
        
        <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
          <VStack align="start" spacing={4}>
            <HStack>
              <Icon as={FiUsers} boxSize={6} color="green.500" />
              <Heading size="md">Person Tracking</Heading>
            </HStack>
            <Text>View and manage people being tracked across cameras.</Text>
            <Button as={RouterLink} to="/camera-tracking" colorScheme="green">
              View Tracks
            </Button>
          </VStack>
        </Box>
        
        <Box p={5} shadow="md" borderWidth="1px" borderRadius="md">
          <VStack align="start" spacing={4}>
            <HStack>
              <Icon as={FiSettings} boxSize={6} color="purple.500" />
              <Heading size="md">System Settings</Heading>
            </HStack>
            <Text>Configure tracking parameters and system settings.</Text>
            <Button colorScheme="purple">
              Settings
            </Button>
          </VStack>
        </Box>
      </SimpleGrid>
    </Box>
  );
};

export default Dashboard; 