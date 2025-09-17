// src/context/AuthProvider.tsx
import React, { useEffect, useState } from "react";
import axiosClient from "../api/axiosClient";
import { AuthContext } from "./AuthContext";
import type { AuthContextType } from "./AuthContext";

interface User {
  id: string;
  name: string;
  email: string;
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("token"));
  const [user, setUser] = useState<User | null>(() => {
    const u = localStorage.getItem("user");
    return u ? JSON.parse(u) : null;
  });

  useEffect(() => {
    if (token) localStorage.setItem("token", token);
    else localStorage.removeItem("token");
  }, [token]);

  useEffect(() => {
    if (user) localStorage.setItem("user", JSON.stringify(user));
    else localStorage.removeItem("user");
  }, [user]);

  const login = async (email: string, password: string) => {
    const res = await axiosClient.post("/auth/login", { email, password });
    const { token: returnedToken, user: returnedUser } = res.data;

    setToken(returnedToken);
    setUser({
      id: returnedUser.id,
      name: returnedUser.name,
      email: returnedUser.email,
    });
  };

  const register = async (name: string, email: string, password: string) => {
    await axiosClient.post("/auth/register", { name, email, password });
  };

  const logout = () => {
    setToken(null);
    setUser(null);
  };

  const contextValue: AuthContextType = {
    user,
    token,
    login,
    register,
    logout,
    isAuthenticated: !!token,
  };

  return <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>;
};
