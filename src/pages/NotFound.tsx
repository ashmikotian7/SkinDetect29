import { useLocation } from "react-router-dom";
import { useEffect } from "react";

const NotFound = () => {
  const location = useLocation();

  useEffect(() => {
    console.error(
      "404 Error: User attempted to access non-existent route:",
      location.pathname
    );
  }, [location.pathname]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-black">
      <div className="text-center bg-black/70 p-10 rounded-2xl shadow-lg border border-emerald-600/50">
        <h1 className="text-6xl font-bold mb-4 text-emerald-600">404</h1>
        <p className="text-xl text-white/80 mb-6">
          Oops! Page not found
        </p>
        <a
          href="/"
          className="text-black bg-emerald-600 hover:bg-emerald-700 px-6 py-2 rounded-lg font-semibold transition"
        >
          Return to Home
        </a>
      </div>
    </div>
  );
};

export default NotFound;
