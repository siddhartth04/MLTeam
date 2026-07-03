/** @type {import('next').NextConfig} */
const API = process.env.MLTEAM_API_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // Proxy /api/* to the FastAPI backend so the browser hits a single origin
  // (no CORS in dev, and SSE streams pass straight through).
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
