import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/useAuth";

const Navbar: React.FC = () => {
  const { user, logout, isAuthenticated } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <nav className="bg-[#0f0f1a] shadow-md sticky top-0 z-40 border-b-2 border-purple-600">
      <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
        
        {/* Left side links */}
        <div className="flex items-center gap-6 text-purple-400 font-medium">
          <Link to="/" className="font-bold text-xl text-purple-500">EMS</Link>
          <NavLink to="/events" label="Events" />
          {isAuthenticated && <NavLink to="/dashboard" label="Dashboard" />}
          {isAuthenticated && <NavLink to="/events/create" label="Create Event" />}
        </div>

        {/* Right side controls */}
        <div className="flex items-center gap-4">
          {!isAuthenticated ? (
            <>
              <Link to="/login" className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded">Login</Link>
              <Link to="/register" className="px-3 py-1 border border-purple-500 text-purple-400 hover:bg-purple-700 hover:text-white rounded">Register</Link>
            </>
          ) : (
            <div className="flex items-center gap-3 text-purple-400">
              <span className="text-sm">Hi, {user?.name}</span>
              <button onClick={handleLogout} className="px-3 py-1 bg-purple-700 hover:bg-purple-800 text-white rounded transition-colors duration-200">Logout</button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

// Separate component to apply underline animation
const NavLink: React.FC<{ to: string; label: string }> = ({ to, label }) => (
  <Link
    to={to}
    className="relative after:absolute after:left-0 after:bottom-[-4px] after:w-0 after:h-[2px] after:bg-purple-500 after:transition-all after:duration-300 hover:after:w-full"
  >
    {label}
  </Link>
);

export default Navbar;


