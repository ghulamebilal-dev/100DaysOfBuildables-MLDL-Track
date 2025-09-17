// src/context/ThemeContext.ts
import { createContext } from "react";
import type { ThemeCtx } from "./themeTypes";

export const ThemeContext = createContext<ThemeCtx | undefined>(undefined);


