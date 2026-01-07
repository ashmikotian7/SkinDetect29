"use client";

import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Link } from "react-router-dom";
import {
  Upload as UploadIcon,
  Image as ImageIcon,
  Scan,
  Shield,
  CheckCircle,
  AlertTriangle,
  Menu,
  Download,
  XCircle,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import ChatBot from "@/components/ChatBot";

const navLinks = [
  { label: "Home", to: "/" },
  { label: "About", to: "/about" },
  { label: "Tips", to: "/tips" },
  { label: "More Info", to: "/moreinfo" },
  { label: "Contact", to: "/contact" },
];

interface AnalysisResult {
  predicted_class: string;
  confidence: number;
  mask_available?: boolean;
  mask_image?: string; // ✅ Added for segmentation display
}

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [results, setResults] = useState<AnalysisResult | null>(null);
  const [progress, setProgress] = useState(0);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isSkinValidationFailed, setIsSkinValidationFailed] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null); // Ref for the results section
  const { toast } = useToast();

  // ===============================
  // Skin Image Validation
  // ===============================
  const validateSkinImage = (imageSrc: string): Promise<boolean> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          resolve(true); // If we can't create canvas, skip validation
          return;
        }

        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        try {
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          const data = imageData.data;
          
          let skinPixelCount = 0;
          let totalPixels = data.length / 4;
          
          // Sample pixels (checking every 20th pixel for performance)
          for (let i = 0; i < data.length; i += 80) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // More permissive skin tone range check
            // Check for a broader range of skin tones
            const isSkinTone = (
              // Basic range check for skin tones
              (r > 50 && r < 255) &&
              (g > 40 && g < 240) &&
              (b > 30 && b < 220) &&
              // Check that it's not overly saturated (not too colorful)
              (Math.abs(r - g) < 100) &&
              (Math.abs(g - b) < 100) &&
              (Math.abs(r - b) < 100)
            );
            
            // Additional check for warm tones (common in skin)
            const isWarmTone = (
              (r >= g) && 
              (g >= b) && 
              (r > 80) && 
              ((r - b) > 20)
            );
            
            // Additional check for cool tones (also common in skin)
            const isCoolTone = (
              (r >= g) && 
              (g >= b) && 
              (g > 50) && 
              ((r - b) > 10)
            );
            
            if (isSkinTone && (isWarmTone || isCoolTone)) {
              skinPixelCount++;
            }
          }
          
          // Lower threshold - if at least 2% of sampled pixels match skin characteristics
          const skinPercentage = (skinPixelCount / (totalPixels / 20)) * 100;
          resolve(skinPercentage >= 2);
        } catch (error) {
          // If there's an error in validation, we'll allow the image to proceed
          console.warn("Image validation failed:", error);
          resolve(true);
        }
      };
      
      img.onerror = () => {
        resolve(true); // If image doesn't load, skip validation
      };
      
      img.src = imageSrc;
    });
  };

  // ===============================
  // Handle Image Upload
  // ===============================
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      toast({
        title: "Invalid file type",
        description: "Please upload a JPG or PNG image.",
        variant: "destructive",
      });
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Upload an image smaller than 10MB.",
        variant: "destructive",
      });
      return;
    }

    // Create preview for validation
    const reader = new FileReader();
    reader.onload = async (e) => {
      const imageSrc = e.target?.result as string;
      
      // Validate if it's a skin image
      const isSkinImage = await validateSkinImage(imageSrc);
      
      if (!isSkinImage) {
        setIsSkinValidationFailed(true);
        setPreviewUrl(imageSrc);
        setSelectedFile(file);
        toast({
          title: "Not a Skin Image",
          description: "The uploaded image doesn't appear to be a skin image. Please upload a clear image of skin concern.",
          variant: "destructive",
        });
        return;
      }
      
      // Valid skin image
      setIsSkinValidationFailed(false);
      setSelectedFile(file);
      setPreviewUrl(imageSrc);
      setAnalysisComplete(false);
      setResults(null);
    };
    
    reader.readAsDataURL(file);
  };

  // ===============================
  // Reset Validation
  // ===============================
  const resetValidation = () => {
    setIsSkinValidationFailed(false);
    setSelectedFile(null);
    setPreviewUrl("");
    setAnalysisComplete(false);
    setResults(null);
  };

  // ===============================
  // Send Image to FastAPI
  // ===============================
  const analyzeAI = async () => {
    if (!selectedFile || isSkinValidationFailed) return;
    setIsAnalyzing(true);
    setProgress(0);

    // Simulate progress bar
    const steps = [20, 40, 60, 80, 100];
    for (const step of steps) {
      await new Promise((resolve) => setTimeout(resolve, 200));
      setProgress(step);
    }

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch("https://skindetect29.onrender.com/predict", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Server error");

      const data: AnalysisResult = await res.json();
      setResults(data);
      setAnalysisComplete(true);

      // Save history
      const history = JSON.parse(localStorage.getItem("skinGuardHistory") || "[]");
      history.unshift({
        id: Date.now(),
        date: new Date().toISOString(),
        filename: selectedFile.name,
        result: data,
      });
      localStorage.setItem("skinGuardHistory", JSON.stringify(history.slice(0, 10)));
    } catch (err) {
      console.error(err);
      toast({
        title: "Analysis Failed",
        description: "Could not analyze the image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ===============================
  // PDF Generation Function
  // ===============================
  const generatePDF = async () => {
    if (!results || !resultsRef.current) return;

    try {
      const { jsPDF } = await import("jspdf");
      const html2canvas = (await import("html2canvas")).default;

      const canvas = await html2canvas(resultsRef.current);
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF("p", "mm", "a4");
      const imgWidth = 190;
      const pageHeight = 280;
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
      let heightLeft = imgHeight;
      let position = 10;

      pdf.addImage(imgData, "PNG", 10, position, imgWidth, imgHeight);
      heightLeft -= pageHeight;

      // Add additional pages if needed
      while (heightLeft >= 0) {
        position = heightLeft - imgHeight;
        pdf.addPage();
        pdf.addImage(imgData, "PNG", 10, position, imgWidth, imgHeight);
        heightLeft -= pageHeight;
      }

      // Add header
      pdf.setFontSize(20);
      pdf.setTextColor(40, 167, 69); // Emerald green color
      pdf.text("SkinGuard AI - Analysis Report", 105, 15, { align: "center" });
      
      // Add footer
      pdf.setFontSize(10);
      pdf.setTextColor(100);
      pdf.text(
        `Generated on: ${new Date().toLocaleDateString()}`,
        105,
        285,
        { align: "center" }
      );

      pdf.save(`skin-analysis-report-${Date.now()}.pdf`);
    } catch (error) {
      console.error("Error generating PDF:", error);
      toast({
        title: "Download Failed",
        description: "Could not generate PDF. Please try again.",
        variant: "destructive",
      });
    }
  };

  // ===============================
  // Suggestion Generator
  // ===============================
  const renderSuggestion = () => {
    if (!results) return null;

    if (results.predicted_class === "Cancerous") {
      return (
        <div className="bg-red-900/40 border border-red-500/50 rounded-lg p-4 text-left">
          <div className="flex items-center gap-2 text-red-400 mb-2">
            <AlertTriangle className="h-5 w-5" />
            <span className="font-semibold">Medical Attention Recommended</span>
          </div>
          <p className="text-sm text-gray-300">
            The lesion appears <strong>cancerous</strong>. Please consult a certified
            dermatologist or oncologist for a detailed examination.
          </p>
          <ul className="mt-2 list-disc list-inside text-sm text-gray-400 space-y-1">
            <li>Avoid self-treatment or topical creams without prescription.</li>
            <li>Book an appointment with a skin specialist as soon as possible.</li>
            <li>Monitor changes in color, border, or bleeding.</li>
          </ul>
        </div>
      );
    }

    if (results.predicted_class === "Non-Cancerous") {
      return (
        <div className="bg-emerald-900/40 border border-emerald-500/50 rounded-lg p-4 text-left">
          <div className="flex items-center gap-2 text-emerald-400 mb-2">
            <CheckCircle className="h-5 w-5" />
            <span className="font-semibold">Low Risk Detected</span>
          </div>
          <p className="text-sm text-gray-300">
            The lesion appears <strong>non-cancerous</strong>. However, continue monitoring it
            for any visual changes or irritation.
          </p>
          <ul className="mt-2 list-disc list-inside text-sm text-gray-400 space-y-1">
            <li>Maintain regular skin hygiene and use sunscreen (SPF 30+).</li>
            <li>Stay hydrated and avoid excessive UV exposure.</li>
            <li>If the spot changes in size or color, get a professional checkup.</li>
          </ul>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="min-h-screen flex flex-col relative text-white">
      <div
        className="absolute inset-0 bg-cover bg-center -z-10"
        style={{ backgroundImage: "url('/doctor.jpg')" }}
      />
      <div className="absolute inset-0 bg-black/80 -z-10"></div>

      {/* HEADER */}
      <header className="sticky top-0 z-40 border-b border-emerald-600/50 bg-black/40 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4 flex justify-between items-center">
          <Link
            to="/"
            className="flex items-center gap-2 text-white hover:text-emerald-500 transition-colors"
          >
            <Shield className="h-6 w-6 text-emerald-500" />
            <span className="text-lg md:text-2xl font-bold">SkinGuard AI</span>
          </Link>

          <nav className="hidden md:flex items-center gap-6">
            {navLinks.map((link, idx) => (
              <Link
                key={idx}
                to={link.to}
                className="text-gray-300 hover:text-emerald-500 transition-colors duration-200 border-b-2 border-transparent hover:border-emerald-500 pb-1"
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
          <div className="md:hidden border-t border-emerald-600/30 bg-black/80 backdrop-blur-sm">
            <nav className="flex flex-col px-6 py-4 gap-3 text-white font-semibold">
              {navLinks.map((link, idx) => (
                <Link
                  key={idx}
                  to={link.to}
                  className="hover:text-emerald-500"
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </div>
        )}
      </header>

      {/* MAIN SECTION */}
      <main className="container mx-auto px-6 py-12 flex-grow">
        <div className="max-w-5xl mx-auto space-y-8">
          <div className="text-center space-y-4">
            <h1 className="text-3xl md:text-4xl font-bold text-white">
              AI Skin Analysis
            </h1>
            <p className="text-lg text-emerald-400 max-w-2xl mx-auto">
              Upload a clear image of your skin concern for instant AI-powered analysis
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Upload Card */}
            <Card className="bg-black/60 backdrop-blur-md border border-emerald-600/40 p-6 rounded-2xl shadow-lg">
              <div className="space-y-6">
                <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
                  <UploadIcon className="h-5 w-5 text-emerald-500" />
                  Upload Image
                </h2>

                {!previewUrl ? (
                  <div
                    className="border-2 border-dashed border-emerald-600/40 rounded-lg p-12 text-center cursor-pointer hover:border-emerald-500 transition-colors"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <ImageIcon className="h-16 w-16 mx-auto text-emerald-400" />
                    <p className="mt-4 text-white font-medium">
                      Click to upload image
                    </p>
                    <p className="text-sm text-gray-400">JPG, PNG up to 10MB</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative">
                      <img
                        src={previewUrl}
                        alt="Uploaded skin"
                        className="w-full h-64 object-cover rounded-lg border border-emerald-600/30"
                      />
                      {isAnalyzing && (
                        <div className="absolute inset-0 bg-black/70 flex items-center justify-center rounded-lg">
                          <Scan className="h-10 w-10 text-emerald-500 animate-spin" />
                        </div>
                      )}
                      {isSkinValidationFailed && (
                        <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center rounded-lg">
                          <XCircle className="h-12 w-12 text-red-500 mb-2" />
                          <p className="text-white font-medium text-center px-4">
                            Not recognized as a skin image
                          </p>
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => fileInputRef.current?.click()}
                        variant="outline"
                        size="sm"
                        className="text-white border-emerald-500 hover:bg-emerald-500/20"
                      >
                        Change
                      </Button>
                      {isSkinValidationFailed ? (
                        <Button
                          onClick={resetValidation}
                          className="bg-emerald-500 hover:bg-emerald-600 text-black flex-1"
                        >
                          Try Another Image
                        </Button>
                      ) : (
                        <Button
                          onClick={analyzeAI}
                          disabled={isAnalyzing || analysisComplete}
                          className="bg-emerald-500 hover:bg-emerald-600 text-black flex-1"
                        >
                          <Scan className="mr-2 h-4 w-4" />
                          {isAnalyzing
                            ? "Analyzing..."
                            : analysisComplete
                            ? "Complete"
                            : "Start Analysis"}
                        </Button>
                      )}
                    </div>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            </Card>

            {/* Results Card */}
            <Card className="bg-black/60 backdrop-blur-md border border-emerald-600/40 p-6 rounded-2xl shadow-lg">
              <div className="space-y-6">
                <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
                  <Shield className="h-5 w-5 text-emerald-500" />
                  Analysis Results
                </h2>

                {!selectedFile && (
                  <div className="text-center py-12 text-gray-400">
                    <Scan className="h-16 w-16 mx-auto mb-4 opacity-50" />
                    <p>Upload an image to start analysis</p>
                  </div>
                )}

                {isAnalyzing && (
                  <div className="space-y-4">
                    <p className="text-center text-white">
                      AI Analysis in Progress...
                    </p>
                    <Progress
                      value={progress}
                      className="h-2 bg-gray-700 [&>div]:bg-emerald-500"
                    />
                  </div>
                )}

                {analysisComplete && results && (
                  <div ref={resultsRef} className="space-y-4 text-center">
                    {/* Application Heading for PDF */}
                    <div className="hidden print:block text-center mb-6">
                      <h1 className="text-2xl font-bold text-emerald-500">SkinGuard AI</h1>
                      <p className="text-gray-600">Skin Cancer Detection & Analysis Report</p>
                      <hr className="my-4 border-emerald-500/50" />
                    </div>

                    {results.predicted_class === "Cancerous" ? (
                      <AlertTriangle className="h-10 w-10 mx-auto text-red-500" />
                    ) : (
                      <CheckCircle className="h-10 w-10 mx-auto text-emerald-400" />
                    )}

                    <h3
                      className={`text-2xl font-semibold ${
                        results.predicted_class === "Cancerous"
                          ? "text-red-400"
                          : "text-emerald-400"
                      }`}
                    >
                      Result: {results.predicted_class}
                    </h3>
                    <p className="text-gray-400">
                      Confidence: {(results.confidence * 100).toFixed(2)}%
                    </p>

                    {/* ✅ Display Segmentation Image */}
                    {results.mask_available && results.mask_image && (
                      <div className="mt-4">
                        <h4 className="text-lg text-emerald-400 font-semibold mb-2">
                          Affected Area Highlighted:
                        </h4>
                        <div className="relative inline-block">
                          <img
                            src={previewUrl}
                            alt="Original skin"
                            className="w-64 h-64 object-cover rounded-lg opacity-80"
                          />
                          <img
                            src={results.mask_image}
                            alt="Segmentation Mask"
                            className="w-64 h-64 object-cover rounded-lg absolute top-0 left-0 mix-blend-screen"
                          />
                        </div>
                      </div>
                    )}

                    {/* Suggestions */}
                    {renderSuggestion()}

                    {/* Download PDF Button */}
                    <div className="pt-4">
                      <Button
                        onClick={generatePDF}
                        className="bg-emerald-500 hover:bg-emerald-600 text-white flex items-center gap-2 mx-auto"
                      >
                        <Download className="h-4 w-4" />
                        Download PDF Report
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div>

        {/* =============================== */}
        {/* Segmentation Color Guide */}
        {/* =============================== */}
        <div className="container mx-auto mt-12 p-6 bg-black/70 border border-emerald-600/50 rounded-2xl text-white max-w-5xl">
          <h2 className="text-xl font-semibold mb-4 text-emerald-400">
            Segmentation Color Guide
          </h2>
          <ul className="list-disc list-inside space-y-2 text-sm text-gray-300">
            <li>🟢 <strong>Green background:</strong> Normal (non-cancerous) skin</li>
            <li>🟠 <strong>Orange core:</strong> High model confidence of affected / cancerous area</li>
            <li>🟣 <strong>Violet or pink border:</strong> Transition between affected and normal regions</li>
            <li>⚫ <strong>Dark patches:</strong> Model not confident / background</li>
          </ul>
         
        </div>
      </main>

      {/* FOOTER */}
      <footer className="border-t border-emerald-600/50 bg-black/70 py-8 md:py-12">
        <div className="container mx-auto px-4 text-center text-white/80 text-sm space-y-1">
          <p>&copy; 2025 SkinGuard AI. All rights reserved.</p>
          <p>
            For informational purposes only; consult a healthcare professional
            for medical advice.
          </p>
        </div>
      </footer>

      <ChatBot />
    </div>
  );
};

export default Upload;
