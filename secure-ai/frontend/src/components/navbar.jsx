import React, { useRef, useState, useEffect } from 'react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';
import { useNavigate } from 'react-router-dom';
import { User, LogOut, Menu, X } from 'lucide-react';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { auth } from '../firebase';
import './styling/navbar.css';
import { Box, Flex, HStack, Link, IconButton, useDisclosure, useColorModeValue, Button } from '@chakra-ui/react';
import { Link as RouterLink } from 'react-router-dom';
import { HamburgerIcon, CloseIcon } from '@chakra-ui/icons';

const NavLink = ({ children, to }) => (
  <Link
    as={RouterLink}
    px={2}
    py={1}
    rounded={'md'}
    _hover={{
      textDecoration: 'none',
      bg: useColorModeValue('gray.200', 'gray.700'),
    }}
    to={to}
  >
    {children}
  </Link>
);

const Navbar = () => {
    const navigate = useNavigate();
    const navRef = useRef(null);
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const [user, setUser] = useState(null);
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const { isOpen, onOpen, onClose } = useDisclosure();
    
    useGSAP(() => {
        // Entrance animation for navbar
        gsap.fromTo(navRef.current, 
            { y: '-100%', opacity: 0 }, 
            { 
                duration: 0.8, 
                y: '0', 
                opacity: 1, 
                ease: 'power3.out' 
            }
        );
    }, []);

    useEffect(() => {
        const handleScroll = () => {
            const scrollPosition = window.scrollY;
            const nav = navRef.current;
            
            if (scrollPosition > 50) {
                nav.classList.add('nav-scrolled');
            } else {
                nav.classList.remove('nav-scrolled');
            }
        };

        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
            setUser(currentUser);
        });
        
        return () => unsubscribe();
    }, []);

    const handleClick = (e, targetId) => {
        e.preventDefault();
        const target = document.querySelector(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
            setMobileMenuOpen(false);
        }
    };

    const toggleMobileMenu = () => {
        setMobileMenuOpen(!mobileMenuOpen);
    };

    const navigateToLogin = () => {
        navigate('/login');
        setMobileMenuOpen(false);
    };

    const toggleDropdown = () => {
        setDropdownOpen(!dropdownOpen);
    };

    const handleLogout = async () => {
        try {
            await signOut(auth);
            setDropdownOpen(false);
        } catch (error) {
            console.error("Error signing out: ", error);
        }
    };

    return (
        <Box bg={useColorModeValue('white', 'gray.800')} px={4} boxShadow="sm">
            <Flex h={16} alignItems={'center'} justifyContent={'space-between'}>
                <IconButton
                    size={'md'}
                    icon={isOpen ? <CloseIcon /> : <HamburgerIcon />}
                    aria-label={'Open Menu'}
                    display={{ md: 'none' }}
                    onClick={isOpen ? onClose : onOpen}
                />
                <HStack spacing={8} alignItems={'center'}>
                    <Box fontWeight="bold">Camera Tracking System</Box>
                    <HStack as={'nav'} spacing={4} display={{ base: 'none', md: 'flex' }}>
                        <NavLink to="/">Dashboard</NavLink>
                        <NavLink to="/camera-tracking">Camera Tracking</NavLink>
                    </HStack>
                </HStack>
                <Flex alignItems={'center'}>
                    <Button
                        variant={'solid'}
                        colorScheme={'blue'}
                        size={'sm'}
                        mr={4}
                    >
                        Settings
                    </Button>
                </Flex>
            </Flex>
        </Box>
    );
};

export default Navbar;