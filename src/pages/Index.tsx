import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Upload,
  FileText,
  Shield,
  Clock,
  Users,
  Menu,
  CheckCircle,
  Send,
} from "lucide-react";
import ChatBot from "@/components/ChatBot";

const navLinks = [
  { label: "Home", to: "/" },
  { label: "About", to: "/about" },
  { label: "Tips", to: "/tips" },
  { label: "More Info", to: "/moreinfo" },
  { label: "Contact", to: "/contact" },
];

const Index = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  return (
    <div className="min-h-screen relative text-white bg-black">
      {/* Background Image */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: "url('/doctor.jpg')" }}
      ></div>

      {/* Lighter overlay */}
      <div className="absolute inset-0 bg-black/0"></div> 

      {/* Content */}
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-emerald-600/50 bg-black/80 backdrop-blur-md sticky top-0 z-40">
          <div className="container mx-auto px-4 py-4 flex justify-between items-center">
            {/* Logo */}
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
                  className="text-gray-300 hover:text-emerald-500 transition-colors duration-200 
                             border-b-2 border-transparent hover:border-emerald-500 pb-1"
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

          {/* Mobile Navigation */}
          {isMobileMenuOpen && (
            <div className="md:hidden border-t border-emerald-500/50 bg-black/90 backdrop-blur-sm">
              <nav className="container mx-auto px-4 py-4 space-y-4">
                {navLinks.map((link, idx) => (
                  <Link
                    key={idx}
                    to={link.to}
                    className="block text-gray-300 hover:text-emerald-500 transition-colors duration-200 
                               border-b border-gray-700 pb-2"
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    {link.label}
                  </Link>
                ))}
              </nav>
            </div>
          )}
        </header>

        {/* Hero Section */}
        <section
          className="relative w-full h-screen flex items-center justify-center text-center"
        >
          <div className="absolute inset-0 bg-black/80"></div>

          <div className="relative z-10 max-w-4xl px-6 sm:px-10">
            <h2 className="text-5xl md:text-6xl font-bold text-white leading-tight mb-6">
              Smarter Way to Detect <br />
              <span className="text-emerald-500">Skin Cancer</span>
            </h2>

            <p className="text-lg md:text-xl text-gray-200 leading-relaxed mb-6">
              Upload an image and get instant insights powered by AI.  
              Detect early, act early — because your skin deserves care.
            </p>

            <div className="flex justify-center flex-wrap gap-4 mb-8">
              <Link
                to="/upload"
                className="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold px-5 py-2 rounded-lg transition-transform transform hover:scale-105 flex items-center shadow-lg shadow-emerald-600/40"
              >
                <Upload className="mr-2 h-4 w-4" />
                Start Detection
              </Link>

              <Link
                to="/about"
                className="border border-emerald-600 text-emerald-600 hover:bg-emerald-600 hover:text-white font-semibold px-5 py-2 rounded-lg transition-transform transform hover:scale-105 flex items-center shadow-md shadow-emerald-600/40"
              >
                Learn More
              </Link>
            </div>

            <div className="flex flex-col sm:flex-row justify-center items-center gap-6 text-sm text-gray-300">
              <div className="flex items-center gap-2">
                <Clock className="h-5 w-5 text-emerald-500" />
                <span>Instant Analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-5 w-5 text-emerald-500" />
                <span>Secure Processing</span>
              </div>
              <div className="flex items-center gap-2">
                <Users className="h-5 w-5 text-emerald-500" />
                <span>Personalized Insights</span>
              </div>
            </div>
          </div>
        </section>


        {/* Why Use Section */}
     <section className="py-16 px-6 bg-black/50">
  <div className="container mx-auto text-center space-y-6">
    <h3 className="text-2xl md:text-4xl font-bold text-white">
      Why Choose SkinDetect AI?
    </h3>
    <div className="grid md:grid-cols-3 gap-6">
      {[
        "Early detection of skin conditions with AI precision",
        "High accuracy: 85%+ reliable results",
        "Results delivered instantly, in under 60 seconds",
      ].map((item, idx) => (
        <Card
          key={idx}
          className="p-6 bg-black/60 border border-emerald-600/50"
        >
          <CheckCircle className="h-6 w-6 text-emerald-500 mx-auto mb-3" />
          <p className="text-white">{item}</p>
        </Card>
      ))}
    </div>
  </div>
</section>

{/* Key Benefits Section */}
<section className="py-16 px-6 bg-black/60">
  <div className="container mx-auto text-center space-y-6">
    <h3 className="text-2xl md:text-4xl font-bold text-white">
      Key Benefits for Your Skin
    </h3>
    <p className="text-white/90 text-lg md:text-xl">
      Take control of your skin wellness with insights designed to help you stay healthy and confident every day.
    </p>
    <div className="grid md:grid-cols-2 gap-6">
      {[
        "Understand potential risk factors for better prevention",
        "Receive actionable advice for daily skin care",
        "Track changes over time to spot trends early",
        "Make informed decisions about professional care",
      ].map((item, idx) => (
        <div
          key={idx}
          className="p-6 bg-black/60 border border-emerald-600/50 rounded-2xl shadow-md flex items-start gap-3 text-left"
        >
          <span className="text-emerald-500 font-bold text-2xl mt-1">✔</span>
          <p className="text-white/90 text-lg">{item}</p>
        </div>
      ))}
    </div>
  </div>
</section>

{/* How to Use Section */}
<section className="py-16 px-6 bg-black/60">
  <div className="container mx-auto text-center space-y-6">
    <h3 className="text-2xl md:text-4xl font-bold text-white">
      How to Use SkinDetect AI
    </h3>
    <div className="grid md:grid-cols-3 gap-6">
      {[
        {
          icon: FileText,
          title: "Upload Your Skin Image",
          desc: "Choose a clear image from your gallery. Ensure the affected area is centered and well-lit.",
        },
        {
          icon: Send,
          title: "AI Analysis",
          desc: "Our AI carefully examines the image to identify potential skin health concerns.",
        },
        {
          icon: FileText,
          title: "View Your Results",
          desc: "Receive a detailed risk assessment along with helpful care suggestions in under a minute.",
        },
      ].map((step, idx) => (
        <Card
          key={idx}
          className="p-6 bg-black/60 border border-emerald-600/50 rounded-2xl shadow-md"
        >
          <step.icon className="h-10 w-10 text-emerald-500 mx-auto mb-3" />
          <h4 className="text-lg font-semibold text-white mb-2">
            {step.title}
          </h4>
          <p className="text-white/90">{step.desc}</p>
        </Card>
      ))}
    </div>
  </div>
</section>


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

        {/* Chatbot */}
        <ChatBot />
      </div>
    </div>
  );
};

export default Index;
