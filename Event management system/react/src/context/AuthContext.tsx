// src/context/AuthContext.ts
import { createContext } from "react";

interface User {
  id: string;
  name: string;
  email: string;
}

export interface AuthContextType {
  user: User | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (name: string, email: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
}

// Just the context — no components here
export const AuthContext = createContext<AuthContextType | undefined>(undefined);
