"use client";

import { useState } from "react";
import { signIn } from "next-auth/react";
import { useRouter } from "next/navigation";

export default function SignIn() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [accessCode, setAccessCode] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    const result = await signIn("credentials", {
      username,
      password,
      accessCode,
      redirect: false,
    });

    setLoading(false);

    if (result?.error) {
      setError("Invalid credentials or access code.");
    } else {
      router.push("/");
    }
  }

  return (
    <div
      className="min-h-screen flex items-center justify-center p-6"
      style={{ background: "var(--paper)", color: "var(--ink)" }}
    >
      <div className="w-full max-w-sm">
        {/* Brand */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-5">
            <div
              className="w-9 h-9 rounded-[7px] flex items-center justify-center flex-shrink-0"
              style={{ background: "var(--accent)" }}
            >
              <span
                className="font-serif text-lg leading-none"
                style={{ color: "var(--paper)" }}
              >
                AI
              </span>
            </div>
            <div>
              <h1
                className="font-serif text-[18px] tracking-[-0.01em]"
                style={{ color: "var(--ink)" }}
              >
                Enterprise AI Gateway
              </h1>
              <p
                className="font-mono text-[10px] uppercase tracking-[.06em]"
                style={{ color: "var(--ink-4)" }}
              >
                Enterprise GenAI Architecture
              </p>
            </div>
          </div>
          <p
            className="text-[13px] leading-relaxed"
            style={{ color: "var(--ink-3)" }}
          >
            Privacy-aware enterprise document intelligence — RAG, hybrid
            inference routing, and semantic response caching.
          </p>
        </div>

        {/* Demo credentials */}
        <div
          className="rounded-lg px-4 py-3 mb-5"
          style={{
            borderLeft: "3px solid var(--accent-mid)",
            background: "var(--accent-light)",
          }}
        >
          <p
            className="font-mono text-[10px] uppercase tracking-[.08em] mb-2"
            style={{ color: "var(--accent)" }}
          >
            Demo credentials
          </p>
          <div className="grid grid-cols-2 gap-1 text-[12px] mb-2">
            <span style={{ color: "var(--ink-3)" }}>Username</span>
            <span className="font-mono" style={{ color: "var(--ink)" }}>
              demo
            </span>
            <span style={{ color: "var(--ink-3)" }}>Password</span>
            <span className="font-mono" style={{ color: "var(--ink)" }}>
              demo123
            </span>
          </div>
          <p className="text-[11px]" style={{ color: "var(--ink-3)" }}>
            Access code (time-limited):{" "}
            <a
              href="https://linkedin.com/in/faizfaruqi"
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:opacity-70"
              style={{
                color: "var(--accent)",
                borderBottom: "1px solid rgba(29,78,137,0.3)",
              }}
            >
              connect on LinkedIn →
            </a>
          </p>
        </div>

        {/* Form */}
        <div
          className="rounded-xl p-6"
          style={{
            border: "1px solid var(--rule)",
            background: "#fff",
          }}
        >
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                className="block font-mono text-[10px] uppercase tracking-[.08em] mb-1.5"
                style={{ color: "var(--ink-4)" }}
              >
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
                autoComplete="username"
                placeholder="demo"
                className="w-full rounded-lg px-3 py-2.5 text-[13px] focus:outline-none focus:ring-2 transition-all"
                style={{
                  border: "1px solid var(--rule)",
                  background: "var(--paper)",
                  color: "var(--ink)",
                }}
              />
            </div>

            <div>
              <label
                className="block font-mono text-[10px] uppercase tracking-[.08em] mb-1.5"
                style={{ color: "var(--ink-4)" }}
              >
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                placeholder="••••••••"
                className="w-full rounded-lg px-3 py-2.5 text-[13px] focus:outline-none focus:ring-2 transition-all"
                style={{
                  border: "1px solid var(--rule)",
                  background: "var(--paper)",
                  color: "var(--ink)",
                }}
              />
            </div>

            <div>
              <label
                className="block font-mono text-[10px] uppercase tracking-[.08em] mb-1.5"
                style={{ color: "var(--ink-4)" }}
              >
                Access Code
              </label>
              <input
                type="text"
                value={accessCode}
                onChange={(e) => setAccessCode(e.target.value)}
                required
                placeholder="Provided by Faiz"
                className="w-full rounded-lg px-3 py-2.5 text-[13px] focus:outline-none focus:ring-2 transition-all"
                style={{
                  border: "1px solid var(--rule)",
                  background: "var(--paper)",
                  color: "var(--ink)",
                }}
              />
            </div>

            {error && (
              <div
                className="flex items-center gap-2 rounded-lg px-3 py-2 text-[12px]"
                style={{
                  background: "var(--coral-light)",
                  border: "1px solid rgba(153,60,29,0.25)",
                  color: "var(--coral)",
                }}
              >
                <svg
                  className="w-3.5 h-3.5 flex-shrink-0"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-lg font-mono text-[11px] uppercase tracking-[.05em] py-2.5 transition-opacity disabled:opacity-40 disabled:cursor-not-allowed"
              style={{
                background: "var(--accent)",
                color: "var(--paper)",
              }}
            >
              {loading ? "Signing in…" : "Sign in →"}
            </button>
          </form>
        </div>

        <p
          className="text-center font-mono text-[10px] uppercase tracking-[.06em] mt-6"
          style={{ color: "var(--ink-4)" }}
        >
          FastAPI · pgvector · Redis · OpenRouter · RAG
        </p>
      </div>
    </div>
  );
}
