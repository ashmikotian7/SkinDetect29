import React, { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Shield, Menu, Mail, User, MessageSquare, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import ChatBot from "@/components/ChatBot";

const navLinks = [
  { label: "Home", to: "/" },
  { label: "About", to: "/about" },
  { label: "Tips", to: "/tips" },
  { label: "More Info", to: "/moreinfo" },
  { label: "Contact", to: "/contact" },
];

const Contact: React.FC = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [message, setMessage] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [result, setResult] = useState("");

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setResult("Sending...");

    const formData = new FormData();
    formData.append("name", fullName);
    formData.append("email", email);
    formData.append("message", message);
    formData.append("access_key", "8c159bf8-edb5-4f31-862f-11deecd08b93"); // Replace with your Web3Forms key

    try {
      const response = await fetch("https://api.web3forms.com/submit", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult("✅ Form submitted successfully!");
        setSubmitted(true);
        setFullName("");
        setEmail("");
        setMessage("");
        setTimeout(() => setSubmitted(false), 4000);
      } else {
        setResult(`Error: ${data.message}`);
        console.log("Web3Forms Error:", data);
      }
    } catch (error) {
      console.log("Fetch error:", error);
      setResult("An error occurred. Please try again.");
    }
  };

  return (
    <div className="min-h-screen relative text-white">
      {/* Background */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: "url('/doctor.jpg')" }}
      />
      <div className="absolute inset-0 bg-black/80" />

      {/* Content */}
      <div className="relative z-10 flex flex-col min-h-screen">
        {/* Header */}
        <header className="border-b border-emerald-600/50 bg-black/40 backdrop-blur-sm sticky top-0 z-40">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <Shield className="h-6 w-6 md:h-8 md:w-8 text-emerald-500" />
              <h1 className="text-lg md:text-2xl font-bold text-white">
                SkinDetect AI
              </h1>
            </div>

            {/* Desktop Nav */}
            <nav className="hidden md:flex items-center gap-6">
              {navLinks.map((link, idx) => (
                <Link
                  key={idx}
                  to={link.to}
                  className={`${
                    link.to === "/contact"
                      ? "text-emerald-500 font-semibold"
                      : "text-gray-200 hover:text-emerald-500"
                  } transition`}
                >
                  {link.label}
                </Link>
              ))}
            </nav>

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden text-white"
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            >
              <Menu className="h-5 w-5" />
            </Button>
          </div>

          {/* Mobile Nav */}
          {isMobileMenuOpen && (
            <div className="md:hidden border-t border-emerald-600/50 bg-black/90">
              <nav className="container mx-auto px-4 py-4 space-y-3">
                {navLinks.map((link, idx) => (
                  <Link
                    key={idx}
                    to={link.to}
                    className="block text-gray-200 hover:text-emerald-500 py-2"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    {link.label}
                  </Link>
                ))}
                <Link
                  to="/upload"
                  className="block px-4 py-3 rounded-md bg-emerald-500 text-black font-semibold text-center hover:bg-emerald-600 transition"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  Get Started
                </Link>
              </nav>
            </div>
          )}
        </header>

        {/* Contact Form Section */}
        <main className="flex-grow flex justify-center items-center px-4 py-16">
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="w-full max-w-2xl p-8 bg-black/70 backdrop-blur-md border border-emerald-600/50 rounded-2xl shadow-xl"
          >
            <h2 className="text-3xl md:text-4xl font-bold text-center text-white mb-8">
              Contact SkinDetect AI
            </h2>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-emerald-400 mb-2 flex items-center gap-2">
                  <User size={18} /> Full Name
                </label>
                <input
                  type="text"
                  name="name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                  placeholder="Enter your full name"
                  className="w-full px-4 py-3 rounded-lg border border-emerald-600/50 bg-black/40 text-white placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-emerald-400 mb-2 flex items-center gap-2">
                  <Mail size={18} /> Email
                </label>
                <input
                  type="email"
                  name="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="your@email.com"
                  className="w-full px-4 py-3 rounded-lg border border-emerald-600/50 bg-black/40 text-white placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 outline-none"
                />
              </div>

              <div>
                <label className="block text-emerald-400 mb-2 flex items-center gap-2">
                  <MessageSquare size={18} /> Message
                </label>
                <textarea
                  name="message"
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  required
                  rows={4}
                  placeholder="Write your message..."
                  className="w-full px-4 py-3 rounded-lg border border-emerald-600/50 bg-black/40 text-white placeholder-gray-400 focus:ring-2 focus:ring-emerald-500 outline-none resize-none"
                />
              </div>

              <motion.button
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                type="submit"
                className="w-full flex items-center justify-center gap-2 py-3 px-6 bg-emerald-500 text-black font-semibold rounded-lg shadow-lg hover:bg-emerald-600 transition"
              >
                <Send size={18} /> Send Message
              </motion.button>

              {result && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-green-400 text-center font-medium mt-4"
                >
                  {result}
                </motion.p>
              )}
            </form>
          </motion.div>
        </main>

        {/* Footer */}
        <footer className="border-t border-emerald-600/50 bg-black/70 py-8 md:py-12">
          <div className="container mx-auto px-4 text-center text-white/80 text-sm">
            <p>
              &copy; 2025 SkinDetect AI. All rights reserved. <br />
              For informational purposes only; consult a healthcare professional for medical advice.
            </p>
          </div>
        </footer>

        {/* Chatbot */}
        <ChatBot />
      </div>
    </div>
  );
};

export default Contact;
