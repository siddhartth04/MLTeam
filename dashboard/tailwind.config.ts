import type { Config } from "tailwindcss";

/**
 * Indigo control-room design tokens. A deep navy/charcoal base, electric
 * indigo/violet accent, and neon-ish semantic colors for pass/fail/pending.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          900: "#080a16", // page background
          850: "#0b0e1f",
          800: "#0f1428", // panel
          700: "#151b33", // raised panel
          600: "#1c2440", // border-ish
          500: "#28324f",
        },
        ink: {
          DEFAULT: "#e8ecf8",
          muted: "#9aa6c8",
          faint: "#5f6b8f",
        },
        accent: {
          DEFAULT: "#7c5cff", // electric indigo
          bright: "#9d84ff",
          dim: "#5a3fd6",
          glow: "#7c5cff",
        },
        cyan: { DEFAULT: "#31d0ff", dim: "#1a92c4" },
        ok: { DEFAULT: "#38e8a0", dim: "#1f9e6c" },
        warn: { DEFAULT: "#ffb648", dim: "#c98420" },
        danger: { DEFAULT: "#ff5d73", dim: "#c33a4e" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        glass: "0 8px 40px -12px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.04)",
        glow: "0 0 24px -2px var(--tw-shadow-color)",
        "glow-lg": "0 0 48px -4px var(--tw-shadow-color)",
      },
      backgroundImage: {
        "accent-grad": "linear-gradient(135deg, #7c5cff 0%, #31d0ff 100%)",
        "panel-grad": "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0))",
      },
      keyframes: {
        "pulse-ring": {
          "0%": { transform: "scale(0.8)", opacity: "0.7" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
        "flow-dash": { to: { strokeDashoffset: "-24" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "fade-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        breathe: {
          "0%, 100%": { opacity: "0.55" },
          "50%": { opacity: "1" },
        },
      },
      animation: {
        "pulse-ring": "pulse-ring 1.8s ease-out infinite",
        "flow-dash": "flow-dash 1s linear infinite",
        shimmer: "shimmer 2s infinite",
        "fade-up": "fade-up 0.4s ease-out both",
        breathe: "breathe 2.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
