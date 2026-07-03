import type { Metadata } from "next";
import type { ReactNode } from "react";

import { Shell } from "@/components/Shell";

import "./globals.css";

export const metadata: Metadata = {
  title: "Mission Control · AI ML Engineering Team",
  description:
    "Live control room for an autonomous ML engineering team — watch agents profile, train, critique, and ship.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
