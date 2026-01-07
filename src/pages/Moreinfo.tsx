import { useState } from "react";
import { Link } from "react-router-dom";
import { Shield, FileText, CheckCircle, Menu, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import ChatBot from "@/components/ChatBot";

const navLinks = [
  { label: "Home", to: "/" },
  { label: "About", to: "/about" },
  { label: "Tips", to: "/tips" },
  { label: "More Info", to: "/moreinfo" },
  { label: "Contact", to: "/contact" },
];

const MoreInfo = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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
              <Shield className="h-6 w-6 md:h-8 md:w-8 text-emerald-600" />
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
                  className="text-gray-300 hover:text-emerald-600 transition-colors duration-200 border-b-2 border-transparent hover:border-emerald-600 pb-1"
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
            <div className="md:hidden border-t border-emerald-600/50 bg-black/70">
              <nav className="container mx-auto px-4 py-4 space-y-3">
                {navLinks.map((link, idx) => (
                  <Link
                    key={idx}
                    to={link.to}
                    className={`block py-2 ${
                      link.to === "/moreinfo"
                        ? "text-emerald-600 font-semibold"
                        : "text-gray-200 hover:text-emerald-600"
                    }`}
                    onClick={() => setIsMobileMenuOpen(false)}
                  >
                    {link.label}
                  </Link>
                ))}
              </nav>
            </div>
          )}
        </header>

        {/* Main Content */}
        <main className="flex-grow container mx-auto px-4 py-16 space-y-16">
          {/* Intro */}
          <section className="text-center max-w-3xl mx-auto space-y-4">
            <h2 className="text-2xl md:text-3xl font-semibold text-emerald-600">
              Why Choose SkinDetect AI?
            </h2>
            <p className="text-white/80">
              Our AI-powered detection system helps with early detection of skin cancer. 
              Trained on thousands of medical images, it provides reliable insights 
              while protecting your privacy.
            </p>
          </section>

          {/* Features */}
          <section className="max-w-4xl mx-auto space-y-6 text-center">
            <h2 className="text-2xl md:text-3xl font-bold text-emerald-600">
              Skin Cancer Research
            </h2>
            <p className="text-white/80">
              Advances in AI and dermatology are improving early detection and
              treatment outcomes. Key studies include:
            </p>
            <div className="grid md:grid-cols-2 gap-4 justify-items-center">
              {[
                {
                  title: "Dermatologist-level classification of skin cancer with deep neural networks",
                  link: "https://www.nature.com/articles/nature21056"
                },
                {
                  title: "Artificial intelligence in dermatology: past, present, and future",
                  link: "https://www.thelancet.com/journals/landig/article/PIIS2589-7500(19)30135-8/fulltext"
                },
                {
                  title: "Evaluation of artificial intelligence–based detection of melanoma",
                  link: "https://jamanetwork.com/journals/jamadermatology/fullarticle/2764608"
                },
                {
                  title: "Deep learning systems for melanoma detection",
                  link: "https://pubmed.ncbi.nlm.nih.gov/33222116/"
                },
              ].map((paper, idx) => (
                <a
                  key={idx}
                  href={paper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-full md:w-auto max-w-md p-4 bg-black/60 border border-emerald-600/50 rounded-lg shadow hover:bg-black/80 transition text-emerald-600 text-sm text-center"
                >
                  {paper.title}
                </a>
              ))}
            </div>
          </section>

          {/* More About Skin Cancer */}
          <section className="max-w-4xl mx-auto space-y-6">
            <h2 className="text-2xl md:text-3xl font-bold text-emerald-600 text-center">
              More About Skin Cancer
            </h2>
            <div className="bg-black/60 p-6 rounded-xl shadow-lg border border-emerald-600/50 space-y-4">
              <p className="text-white/80">
                Skin cancer is the most common form of cancer worldwide. It
                occurs when skin cells grow uncontrollably due to DNA damage,
                often from UV exposure.
              </p>
              <ul className="list-disc pl-6 space-y-2 text-emerald-500 text-sm">
                <li>
                  <strong>Types:</strong> Basal cell carcinoma, squamous cell
                  carcinoma, and melanoma.
                </li>
                <li>
                  <strong>Risk Factors:</strong> Excessive sun exposure, tanning
                  beds, family history, fair skin.
                </li>
                <li>
                  <strong>Prevention:</strong> Sunscreen, protective clothing,
                  avoiding peak UV hours, regular self-checks.
                </li>
              </ul>
              <p className="text-white/80">
                Early detection greatly increases the chances of successful
                treatment. Regular skin checks and dermatology visits are
                strongly recommended.
              </p>
            </div>

            {/* Did You Know Card */}
            <div className="mt-6 bg-emerald-600/20 border border-emerald-500 rounded-lg p-5 flex items-start gap-3">
              <Info className="h-6 w-6 text-emerald-500 mt-1" />
              <p className="text-emerald-500 text-sm">
                Did you know? Skin cancer is highly preventable - experts estimate
                that <strong>over 90% of cases are linked to UV exposure</strong> and can be
                reduced with consistent sun protection.
              </p>
            </div>
          </section>

          {/* CTA */}
          <section className="text-center space-y-6">
            <h2 className="text-2xl md:text-3xl font-bold text-emerald-600">
              Ready to Experience the Future of Skin Health?
            </h2>
            <p className="text-white/80 max-w-xl mx-auto">
              Start your free analysis today. Upload a photo of your skin concern 
              and let our AI provide insights instantly.
            </p>
          </section>
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

        {/* ChatBot */}
        <ChatBot />
      </div>
    </div>
  );
};

export default MoreInfo;
