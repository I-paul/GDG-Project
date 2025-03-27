import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

// Smooth scroll to sections
export const setupSmoothScroll = () => {
    document.querySelectorAll(".scroll-to").forEach((button) => {
        button.addEventListener("click", (e) => {
            e.preventDefault();
            const targetId = button.getAttribute("href").substring(1);
            const targetSection = document.getElementById(targetId);

            gsap.to(window, {
                scrollTo: { y: targetSection, offsetY: 80 },
                duration: 1.5,
                ease: "power2.out",
            });
        });
    });
};

// Fade-in animation for sections on scroll
export const setupSectionFadeIn = () => {
    gsap.utils.toArray("section").forEach((section) => {
        gsap.fromTo(section, 
            { opacity: 0, y: 50 }, 
            { opacity: 1, y: 0, duration: 1, ease: "power2.out", 
              scrollTrigger: {
                  trigger: section,
                  start: "top bottom", 
                  toggleActions: "play none none reverse",
              }
            }
        );
    });
};

// Animation for camera list and form in AddCam component
export const setupAddCamAnimations = (addCamRef, formRef, cameraListRef) => {
    if (!addCamRef?.current) console.warn("addCamRef is null or undefined.");
    if (!formRef?.current) console.warn("formRef is null or undefined.");
    if (!cameraListRef?.current) console.warn("cameraListRef is null or undefined.");

    if (!addCamRef?.current || !formRef?.current || !cameraListRef?.current) {
        console.warn("One or more animation targets are missing.");
        return;
    }

    // Animate the camera list
    if (cameraListRef.current) {
        gsap.fromTo(
            cameraListRef.current,
            { opacity: 0, y: 50, visibility: "hidden" },
            {
                opacity: 1,
                y: 0,
                visibility: "visible",
                duration: 1.5,
                ease: "power3.out",
                scrollTrigger: {
                    trigger: addCamRef.current,
                    start: "top 80%",
                    end: "top 30%",
                    scrub: true,
                    toggleActions: "play none none reverse",
                },
            }
        );
    }

    // Animate the form
    if (formRef.current) {
        gsap.fromTo(
            formRef.current,
            { opacity: 0, y: 50, visibility: "hidden" },
            {
                opacity: 1,
                y: 0,
                visibility: "visible",
                duration: 1.5,
                ease: "power3.out",
                scrollTrigger: {
                    trigger: addCamRef.current,
                    start: "top 80%",
                    end: "top 30%",
                    scrub: true,
                    toggleActions: "play none none reverse",
                },
            }
        );
    }
};