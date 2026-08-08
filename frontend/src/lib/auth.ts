import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";

const DEMO_USERNAME = process.env.DEMO_USERNAME || "demo";
const DEMO_PASSWORD = process.env.DEMO_PASSWORD || "demo123";

// Server-side only (this callback never runs in the browser) — reuses the
// same backend base URL the client-side API wrapper uses, just read
// directly from process.env rather than through lib/api.ts.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function isAccessCodeValid(code: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/auth/verify-code`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    return data.valid === true;
  } catch {
    // Backend/Redis unreachable — deny sign-in rather than admit everyone.
    // Unlike the rate limiter or budget tracker, an auth check must fail
    // closed: staying permissive during an outage is worse than a
    // temporarily unavailable demo.
    return false;
  }
}

export const authOptions: NextAuthOptions = {
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        username: { label: "Username", type: "text" },
        password: { label: "Password", type: "password" },
        accessCode: { label: "Access Code", type: "text" },
      },
      async authorize(credentials) {
        const validUser =
          credentials?.username === DEMO_USERNAME &&
          credentials?.password === DEMO_PASSWORD;
        if (!validUser) return null;

        const validCode = await isAccessCodeValid(credentials?.accessCode ?? "");
        if (!validCode) return null;

        return {
          id: "1",
          name: "Demo User",
          email: "demo@hybrid-ai.demo",
        };
      },
    }),
  ],
  pages: {
    signIn: "/auth/signin",
  },
  session: {
    strategy: "jwt",
    maxAge: 24 * 60 * 60,
  },
  secret: process.env.NEXTAUTH_SECRET,
};
