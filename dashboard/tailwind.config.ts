import type { Config } from "tailwindcss";

/**
 * Light, professional design tokens. A calm off-white canvas, crisp white
 * cards, a confident indigo accent, and clear semantic colors for
 * pass/fail/pending. Token NAMES are preserved from the previous theme so
 * component markup is unchanged — only the values (and their light/dark
 * intent) flip: higher `base` numbers are the recessed/page surfaces, lower
 * numbers are the subtle raised fills and borders.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          900: "#f5f6f8", // page background (recessed canvas)
          850: "#f8f9fb", // sidebar
          800: "#ffffff", // panel / card surface
          700: "#f1f3f6", // raised fill / hover / chip
          600: "#e4e7ec", // borders
          500: "#d5d9e0", // stronger borders / raised outline
        },
        ink: {
          DEFAULT: "#1a2231", // primary text (dark slate)
          muted: "#586173", // secondary text
          faint: "#98a1b0", // tertiary text / hints
        },
        accent: {
          DEFAULT: "#4f46e5", // indigo-600 — button fills (white text passes AA)
          bright: "#4338ca", // indigo-700 — text emphasis on light surfaces
          dim: "#6366f1", // indigo-500 — gradient / subtle
          glow: "#6366f1",
        },
        cyan: { DEFAULT: "#0891b2", dim: "#0e7490" },
        ok: { DEFAULT: "#059669", dim: "#047857" },
        warn: { DEFAULT: "#d97706", dim: "#b45309" },
        danger: { DEFAULT: "#dc2626", dim: "#b91c1c" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        // Clean, layered elevation tuned for a light canvas.
        glass:
          "0 1px 2px rgba(16,24,40,0.04), 0 6px 20px -8px rgba(16,24,40,0.10)",
        card: "0 1px 2px rgba(16,24,40,0.05), 0 1px 3px rgba(16,24,40,0.06)",
        glow: "0 8px 24px -8px var(--tw-shadow-color)",
        "glow-lg": "0 16px 40px -12px var(--tw-shadow-color)",
      },
      backgroundImage: {
        "accent-grad": "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
        "panel-grad": "linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0))",
      },
      keyframes: {
        "pulse-ring": {
          "0%": { transform: "scale(0.8)", opacity: "0.55" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
        "flow-dash": { to: { strokeDashoffset: "-24" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "fade-up": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        breathe: {
          "0%, 100%": { opacity: "0.5" },
          "50%": { opacity: "0.85" },
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
