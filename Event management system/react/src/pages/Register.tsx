// // src/pages/Register.tsx
// import React, { useState, useEffect } from "react";
// import { useNavigate } from "react-router-dom";
// import { useAuth } from "../context/useAuth";
// import axios from "axios";

// const Register: React.FC = () => {
//   const { register } = useAuth();
//   const navigate = useNavigate();

//   const [name, setName] = useState("");
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   const [err, setErr] = useState("");
//   const [showForm, setShowForm] = useState(false); // For animation on load

//   useEffect(() => {
//     // Animate form after mount
//     setTimeout(() => setShowForm(true), 100);
//   }, []);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     setErr("");
//     try {
//       await register(name, email, password);
//       navigate("/login");
//     } catch (error: unknown) {
//       if (axios.isAxiosError(error)) {
//         setErr(error.response?.data?.message || "Registration failed");
//       } else if (error instanceof Error) {
//         setErr(error.message);
//       } else {
//         setErr("Registration failed");
//       }
//     }
//   };

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black flex items-center justify-center px-4 py-12">
//       <form
//         onSubmit={handleSubmit}
//         className={`w-full max-w-md transform transition-all duration-700 ease-in-out ${
//           showForm ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
//         } bg-gray-900 text-white p-8 rounded-xl shadow-lg shadow-black/30 border border-gray-700`}
//       >
//         <h2 className="text-3xl font-bold mb-6 text-center text-purple-400">
//           Create an Account
//         </h2>

//         {err && (
//           <div className="mb-4 text-red-400 text-sm text-center border border-red-500 rounded p-2 bg-red-900/20">
//             {err}
//           </div>
//         )}

//         <input
//           value={name}
//           onChange={(e) => setName(e.target.value)}
//           placeholder="Full Name"
//           className="w-full mb-4 p-3 rounded-lg bg-gray-800 border border-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
//         />
//         <input
//           value={email}
//           onChange={(e) => setEmail(e.target.value)}
//           placeholder="Email"
//           type="email"
//           className="w-full mb-4 p-3 rounded-lg bg-gray-800 border border-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
//         />
//         <input
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//           placeholder="Password"
//           type="password"
//           className="w-full mb-6 p-3 rounded-lg bg-gray-800 border border-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
//         />

//         <button
//           type="submit"
//           className="w-full py-3 bg-purple-600 hover:bg-purple-700 transition-all duration-300 rounded-lg font-semibold tracking-wide text-white shadow-md hover:shadow-xl"
//         >
//           Create Account
//         </button>
//       </form>
//     </div>
//   );
// };

// export default Register;




// src/pages/Register.tsx
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/useAuth";
import axios from "axios";

// ✅ Password validation function
const validatePassword = (password: string): string | null => {
  if (password.length < 8) {
    return "Password must be at least 8 characters long.";
  }
  if (password.length > 20) {
    return "Password must not exceed 20 characters.";
  }
  if (!/[A-Z]/.test(password)) {
    return "Password must include at least one uppercase letter.";
  }
  if (!/[a-z]/.test(password)) {
    return "Password must include at least one lowercase letter.";
  }
  if (!/[0-9]/.test(password)) {
    return "Password must include at least one digit.";
  }
  if (!/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
    return "Password must include at least one special character.";
  }
  return null;
};

const Register: React.FC = () => {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState(false); // For animation on load

  useEffect(() => {
    // Animate form after mount
    setTimeout(() => setShowForm(true), 100);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErr("");

    const pwdError = validatePassword(password);
    if (pwdError) {
      setPasswordError(pwdError);
      return;
    } else {
      setPasswordError(null);
    }

    try {
      await register(name, email, password);
      navigate("/login");
    } catch (error: unknown) {
      if (axios.isAxiosError(error)) {
        setErr(error.response?.data?.message || "Registration failed");
      } else if (error instanceof Error) {
        setErr(error.message);
      } else {
        setErr("Registration failed");
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black flex items-center justify-center px-4 py-12">
      <form
        onSubmit={handleSubmit}
        className={`w-full max-w-md transform transition-all duration-700 ease-in-out ${
          showForm ? "opacity-100 translate-y-0" : "opacity-0 translate-y-10"
        } bg-gray-900 text-white p-8 rounded-xl shadow-lg shadow-black/30 border border-gray-700`}
      >
        <h2 className="text-3xl font-bold mb-6 text-center text-purple-400">
          Create an Account
        </h2>

        {err && (
          <div className="mb-4 text-red-400 text-sm text-center border border-red-500 rounded p-2 bg-red-900/20">
            {err}
          </div>
        )}

        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Full Name"
          className="w-full mb-4 p-3 rounded-lg bg-gray-800 border border-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
        />
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          type="email"
          className="w-full mb-4 p-3 rounded-lg bg-gray-800 border border-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
        />
        <input
          value={password}
          onChange={(e) => {
            const value = e.target.value;
            setPassword(value);
            const validationError = validatePassword(value);
            setPasswordError(validationError);
          }}
          placeholder="Password"
          type="password"
          className="w-full mb-2 p-3 rounded-lg bg-gray-800 border border-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all"
        />
        {/* 🔥 Show password validation error */}
        {passwordError && (
          <div className="mb-4 text-red-400 text-sm">
            {passwordError}
          </div>
        )}

        <button
          type="submit"
          className="w-full py-3 bg-purple-600 hover:bg-purple-700 transition-all duration-300 rounded-lg font-semibold tracking-wide text-white shadow-md hover:shadow-xl disabled:opacity-50"
          disabled={!!passwordError}
        >
          Create Account
        </button>
      </form>
    </div>
  );
};

export default Register;
