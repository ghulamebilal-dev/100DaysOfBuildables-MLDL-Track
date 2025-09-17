// src/pages/Home.tsx
import React, { useEffect, useState } from "react";
import "../styles/Home.css"; // 👈 External CSS

const Home: React.FC = () => {
  const [showContent, setShowContent] = useState(false);

  useEffect(() => {
    setTimeout(() => setShowContent(true), 100);
  }, []);

  return (
    <div className="home-background min-h-screen flex items-center justify-center bg-cover bg-center relative px-4">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black bg-opacity-60 z-0"></div>

      {/* Content */}
      <div
        className={`relative z-10 text-center text-white transition-all duration-700 ${
          showContent ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
        }`}
      >
        <h1 className="text-4xl md:text-5xl font-extrabold text-purple-400 drop-shadow-lg">
          Welcome to Event Management System
        </h1>
        <p className="mt-6 max-w-2xl mx-auto text-lg md:text-xl text-gray-200">
          Create events, RSVP, and manage your events. Sign up and start organizing your events today!
        </p>
      </div>
    </div>
  );
};

export default Home;
