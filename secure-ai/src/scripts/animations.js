import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

// Smooth scroll to sections
document.querySelectorAll(".scroll-to").forEach((button) => {
    button.addEventListener("click", (e) => {
        e.preventDefault();
        const targetId = button.getAttribute("href").substring(1);
        const targetSection = document.getElementById(targetId);

        gsap.to(window, {
            scrollTo: { y: targetSection, offsetY: 80 }, // Adjust offset for fixed headers
            duration: 1.5,
            ease: "power2.out",
        });
    });
});

// Fade-in animation for sections on scroll
gsap.utils.toArray("section").forEach((section) => {
    gsap.from(section, {
        opacity: 0,
        y: 50,
        duration: 1,
        ease: "power2.out",
        scrollTrigger: {
            trigger: section,
            start: "top 80%",
            toggleActions: "play none none reverse",
        },
    });
});