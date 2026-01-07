import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Shield, Menu } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import ChatBot from "@/components/ChatBot";

const navLinks = [
  { label: "Home", to: "/" },
  { label: "About", to: "/about" },
  { label: "Tips", to: "/tips" },
  { label: "More Info", to: "/moreinfo" },
  { label: "Contact", to: "/contact" },
];

const categories = [
  {
    key: "precautions",
    title: "Skin Safety – Precautions",
    slides: [
      {
        title: "Use Sunscreen",
        steps: [
          { title: "Apply generously", description: "Use SPF 30+ and reapply every 2 hours.", duration: "Daily habit" },
          { title: "Cover exposed areas", description: "Don’t forget ears, neck, and hands." },
        ],
      },
      {
        title: "Avoid Peak Sun Hours",
        steps: [{ title: "Stay indoors", description: "Avoid sun between 10 AM – 4 PM." }],
      },
      {
        title: "Protective Clothing",
        steps: [
          { title: "Wear hats & sunglasses", description: "Wide-brim hats and UV-protective glasses." },
          { title: "Long sleeves", description: "Lightweight but protective fabrics help." },
        ],
      },
      {
        title: "Avoid Tanning Beds",
        steps: [{ title: "Skip artificial tanning", description: "UV exposure from tanning beds damages DNA." }],
      },
    ],
  },
  {
    key: "detection",
    title: "Early Detection – Know the Signs",
    slides: [
      {
        title: "Self Check",
        steps: [{ title: "Look for changes", description: "Monitor moles, spots, and skin patches regularly." }],
      },
      {
        title: "ABCDE Rule",
        steps: [
          { title: "Asymmetry", description: "One half looks different from the other." },
          { title: "Border", description: "Irregular, blurred, or jagged edges." },
          { title: "Color", description: "Varied shades (brown, black, red, white)." },
          { title: "Diameter", description: "Larger than 6mm (pencil eraser)." },
          { title: "Evolving", description: "Changes in size, shape, or color." },
        ],
      },
      {
        title: "Seek Help",
        steps: [{ title: "See a dermatologist", description: "If you notice suspicious or changing spots, book a check-up." }],
      },
    ],
  },
];

const stepVariants = {
  hidden: { opacity: 0, y: 15 },
  visible: (i: number) => ({ opacity: 1, y: 0, transition: { delay: i * 0.12, duration: 0.35 } }),
};

const SkinGuardSlides = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [categoryIndex, setCategoryIndex] = useState(0);
  const [expandedSlides, setExpandedSlides] = useState<number[]>([]);
  const currentCategory = categories[categoryIndex];

  const toggleSlide = (index: number) => {
    setExpandedSlides((prev) =>
      prev.includes(index) ? prev.filter((i) => i !== index) : [...prev, index]
    );
  };

  return (
    <div className="min-h-screen flex flex-col relative text-white">
      {/* Background */}
      <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: "url('/doctor.jpg')" }} />
      <div className="absolute inset-0 bg-black/80" />

      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Header */}
        <header className="border-b border-emerald-600/50 bg-black/40 backdrop-blur-sm sticky top-0 z-40">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Shield className="h-6 w-6 md:h-8 md:w-8 text-emerald-600" />
              <h1 className="text-lg md:text-2xl font-bold text-white">SkinDetect AI</h1>
            </div>
            <nav className="hidden md:flex items-center gap-6">
              {navLinks.map((link, idx) => (
                <Link
                  key={idx}
                  to={link.to}
                  className="text-gray-300 hover:text-emerald-600 transition-colors duration-200 border-b-2 border-transparent hover:border-emerald-600 pb-1"
                >
                  {link.label}
                </Link>
              ))}
            </nav>
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden text-white"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              <Menu className="h-5 w-5" />
            </Button>
          </div>

          {isMobileMenuOpen && (
            <div className="md:hidden border-t border-emerald-600/50 bg-black/80 backdrop-blur-sm">
              <nav className="container mx-auto px-4 py-4 space-y-4">
                {navLinks.map((link, idx) => (
                  <Link
                    key={idx}
                    to={link.to}
                    className="block text-gray-300 hover:text-emerald-600 transition-colors duration-200 border-b border-gray-700 pb-2"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    {link.label}
                  </Link>
                ))}
                <Link
                  to="/upload"
                  className="block px-4 py-3 rounded-md bg-emerald-600 text-black font-semibold text-center hover:bg-emerald-700 transition shadow-sm"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Get Started
                </Link>
              </nav>
            </div>
          )}
        </header>

        {/* Main Content */}
        <main className="flex-grow container mx-auto px-4 py-12">
          {/* Category Tabs */}
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {categories.map((cat, idx) => (
              <button
                key={cat.key}
                onClick={() => {
                  setCategoryIndex(idx);
                  setExpandedSlides([]);
                }}
                className={`px-4 py-2 rounded-full text-sm font-semibold transition-colors ${
                  idx === categoryIndex
                    ? "bg-emerald-600 text-black shadow-md"
                    : "bg-black/60 border border-emerald-600/50 text-gray-300 hover:bg-black/80"
                }`}
              >
                {cat.title}
              </button>
            ))}
          </div>

          <motion.h2
            key={currentCategory.key}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="text-2xl md:text-3xl font-bold text-center mb-8 text-emerald-500"
          >
            {currentCategory.title}
          </motion.h2>

          {/* Slides */}
          <div className="w-full max-w-4xl mx-auto flex flex-col gap-4">
            {currentCategory.slides.map((slide, idx) => {
              const isExpanded = expandedSlides.includes(idx);
              return (
                <motion.div
                  key={`${currentCategory.key}-${idx}`}
                  initial={{ opacity: 0, x: 150 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -150 }}
                  transition={{ duration: 0.5 }}
                  className="w-full rounded-lg bg-black/60 border border-emerald-600/50 shadow-md"
                >
                  <div
                    className="p-4 flex items-center justify-between cursor-pointer hover:bg-black/40 transition-colors rounded-t-lg"
                    onClick={() => toggleSlide(idx)}
                  >
                    <div className="flex items-center">
                      <span className="text-xl md:text-2xl font-bold text-emerald-600 mr-3">{`${idx + 1}.`}</span>
                      <h3 className="text-lg md:text-xl font-semibold text-emerald-400">{slide.title}</h3>
                    </div>
                    <span className="text-xl text-emerald-600">{isExpanded ? "▲" : "▼"}</span>
                  </div>

                  <AnimatePresence>
                    {isExpanded && (
                      <motion.div
                        key="steps"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.4 }}
                        className="px-4 pb-4 flex flex-col gap-2"
                      >
                        {slide.steps.map((step, i) => (
                          <motion.div
                            key={i}
                            className="p-3 rounded-lg bg-black/50 shadow-inner border-l-4 border-emerald-600"
                            variants={stepVariants}
                            initial="hidden"
                            animate="visible"
                            custom={i}
                          >
                            <div className="flex items-center mb-1">
                              <div className="w-6 h-6 flex items-center justify-center rounded-full bg-emerald-600 text-black font-bold mr-2 text-sm">
                                {i + 1}
                              </div>
                              <h4 className="text-sm md:text-base font-semibold text-emerald-400">
                                {step.title}
                              </h4>
                            </div>
                            <p className="text-gray-300 text-xs md:text-sm mb-0.5">{step.description}</p>
                            {step.duration && <p className="text-gray-400 text-xs md:text-xs">⏱ {step.duration}</p>}
                          </motion.div>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              );
            })}
          </div>
        </main>

 {/* Footer */}
 <footer className="border-t border-emerald-600/50 bg-black/70 py-8 md:py-12">
  <div className="container mx-auto px-4 text-center text-white/80 text-sm">
    <p>
      &copy; 2025 SkinDetect AI. All rights reserved. 
      <br></br> 
      For informational purposes only; consult a healthcare professional for medical advice.
    </p>
  </div>
</footer>
        <ChatBot />
      </div>
    </div>
  );
};

export default SkinGuardSlides;
