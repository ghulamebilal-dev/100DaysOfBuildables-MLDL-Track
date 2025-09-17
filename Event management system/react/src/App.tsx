// src/App.tsx
import React from "react";
import { ThemeProvider } from "./context/ThemeProvider"; 
import { AuthProvider } from "./context/AuthProvider";
import AppRoutes from "./routes/AppRoutes";

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </ThemeProvider>
  );
};

export default App;

