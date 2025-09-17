// src/context/themeTypes.ts
export type Theme = "light" | "dark";

export interface ThemeCtx {
  theme: Theme;
  toggleTheme: () => void;
}


