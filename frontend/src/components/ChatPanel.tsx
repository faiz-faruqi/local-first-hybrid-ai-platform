"use client";

import { useEffect, useRef, useState } from "react";
import type {
  InferenceProvider,
  ModelInfo,
  QueryProfile,
  QueryResponse,
  RoutingDecision,
  SourceDocument,
} from "@/lib/api";
import { getHealth, getModels, queryDocuments, streamQuery } from "@/lib/api";
import ResponseMeta from "./ResponseMeta";
import SourceDocs from "./SourceDocs";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  provider?: InferenceProvider;
  cached?: boolean;
  latencyMs?: number;
  sources?: SourceDocument[];
  modelAlias?: string | null;
  classification?: QueryProfile | null;
  routingDecision?: RoutingDecision | null;
  error?: boolean;
  // Set when a stream fails mid-flight — unlike `error`, this doesn't
  // replace already-streamed partial content, it annotates alongside it.
  streamError?: string;
}

interface ChatPanelProps {
  onResponse: (provider: InferenceProvider | null) => void;
  onLoading: (loading: boolean) => void;
  onReset?: () => void;
}

const DEPARTMENTS = ["IT", "Sales", "Marketing", "Finance"];

const SAMPLE_QUERIES = [
  "What are the termination conditions in the vendor contracts?",
  "Which contracts have data protection obligations?",
  "What is the total contract value across the portfolio?",
  "What SLA commitments are specified in the agreements?",
];

export default function ChatPanel({ onResponse, onLoading, onReset }: ChatPanelProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [forceCloud, setForceCloud] = useState(false);
  const [streamingEnabled, setStreamingEnabled] = useState(false);
  const [demoMode, setDemoMode] = useState<boolean | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [selectedDepartment, setSelectedDepartment] = useState<string>("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    let alive = true;
    getHealth()
      .then((h) => {
        if (alive) setDemoMode(h.demo_mode);
      })
      .catch(() => {
        if (alive) setDemoMode(false);
      });
    getModels()
      .then((m) => {
        if (alive) setModels(m);
      })
      .catch(() => {
        // Models endpoint unavailable — model selector stays empty (auto-routing only)
      });
    return () => {
      alive = false;
    };
  }, []);

  function handleReset() {
    setMessages([]);
    setInput("");
    setForceCloud(false);
    setSelectedModel("");
    setSelectedDepartment("");
    onResponse(null);
    onLoading(false);
    onReset?.();
    inputRef.current?.focus();
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    onLoading(true);
    onResponse(null);

    if (streamingEnabled) {
      const assistantId = crypto.randomUUID();
      setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content: "" }]);

      await streamQuery(
        {
          query: trimmed,
          top_k: 5,
          force_cloud: forceCloud,
          model: selectedModel || undefined,
          department: selectedDepartment || undefined,
        },
        (delta) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + delta } : m))
          );
        },
        (done) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? {
                    ...m,
                    provider: done.provider,
                    cached: done.cached,
                    latencyMs: done.latency_ms,
                    sources: done.sources,
                    modelAlias: done.model_alias,
                    classification: done.classification,
                    routingDecision: done.routing_decision,
                  }
                : m
            )
          );
          onResponse(done.provider);
        },
        (message) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, streamError: message } : m))
          );
          onResponse(null);
        }
      );

      setLoading(false);
      onLoading(false);
      return;
    }

    try {
      const result: QueryResponse = await queryDocuments({
        query: trimmed,
        top_k: 5,
        force_cloud: forceCloud,
        model: selectedModel || undefined,
        department: selectedDepartment || undefined,
      });

      const assistantMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: result.answer,
        provider: result.provider,
        cached: result.cached,
        latencyMs: result.latency_ms,
        sources: result.sources,
        modelAlias: result.model_alias,
        classification: result.classification,
        routingDecision: result.routing_decision,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      onResponse(result.provider);
    } catch (err: unknown) {
      const errorMsg: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: err instanceof Error ? err.message : "An unexpected error occurred.",
        error: true,
      };
      setMessages((prev) => [...prev, errorMsg]);
      onResponse(null);
    } finally {
      setLoading(false);
      onLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar — always visible so a fresh query is one click away */}
      <div className="flex justify-end mb-2">
        <button
          onClick={handleReset}
          disabled={loading}
          className="font-mono text-xs px-3 py-1 rounded-lg border transition-colors disabled:opacity-40 hover:opacity-80"
          style={{
            borderColor: "var(--rule)",
            background: "var(--paper)",
            color: "var(--ink-3)",
          }}
        >
          ↺ New chat
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-1 py-2 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="pt-4">
            <p className="text-sm mb-4" style={{ color: "var(--ink-3)" }}>
              Ask questions about the sample vendor contract portfolio. The AI Gateway
              will automatically classify your query and route it to the optimal model.
              Try one of these:
            </p>
            <div className="flex flex-col gap-2">
              {SAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-left text-sm px-4 py-3 rounded-lg border transition-colors hover:border-accent"
                  style={{
                    borderColor: "var(--rule)",
                    background: "var(--paper)",
                    color: "var(--ink-2)",
                  }}
                >
                  <span className="font-mono text-xs mr-2" style={{ color: "var(--accent-mid)" }}>→</span>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            {msg.role === "user" ? (
              <div
                className="max-w-[85%] rounded-2xl rounded-tr-md px-4 py-3 text-sm leading-relaxed"
                style={{ background: "var(--accent)", color: "#fff" }}
              >
                {msg.content}
              </div>
            ) : (
              <div className="max-w-[100%] w-full">
                <div
                  className="rounded-2xl rounded-tl-md px-4 py-3 text-sm leading-relaxed border"
                  style={{
                    background: msg.error ? "var(--coral-light)" : "var(--rule-light)",
                    borderColor: msg.error ? "#f5c4b3" : "var(--rule)",
                    color: msg.error ? "var(--coral)" : "var(--ink-2)",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {msg.content}
                </div>
                {msg.streamError && (
                  <p className="mt-1 text-xs" style={{ color: "var(--coral)" }}>
                    ⚠ {msg.streamError}
                  </p>
                )}
                {msg.provider && !msg.error && (
                  <>
                    <ResponseMeta
                      provider={msg.provider}
                      cached={msg.cached ?? false}
                      latencyMs={msg.latencyMs ?? 0}
                      modelAlias={msg.modelAlias}
                      classification={msg.classification}
                      routingDecision={msg.routingDecision}
                    />
                    {msg.sources && <SourceDocs sources={msg.sources} />}
                  </>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && !streamingEnabled && (
          <div className="flex justify-start">
            <div
              className="rounded-2xl rounded-tl-md px-4 py-3 border"
              style={{ background: "var(--rule-light)", borderColor: "var(--rule)" }}
            >
              <div className="flex items-center gap-1.5">
                {[0, 150, 300].map((delay) => (
                  <span
                    key={delay}
                    className="inline-block w-1.5 h-1.5 rounded-full animate-bounce"
                    style={{ background: "var(--ink-4)", animationDelay: `${delay}ms` }}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input area */}
      <div
        className="border-t pt-3 mt-3 space-y-2"
        style={{ borderColor: "var(--rule)" }}
      >
        {/* Model selector (Phase 1) + Force cloud toggle */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Model selector dropdown */}
          {models.length > 0 && (
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs" style={{ color: "var(--ink-4)" }}>
                model:
              </span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={loading}
                className="font-mono text-xs px-2 py-1 rounded-lg border transition-colors disabled:opacity-40"
                style={{
                  borderColor: "var(--rule)",
                  background: "var(--paper)",
                  color: "var(--ink-2)",
                }}
              >
                <option value="">auto (gateway decides)</option>
                {models.map((m) => (
                  <option key={m.alias} value={m.alias}>
                    {m.alias} ({m.tier})
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Department selector — self-reported, for usage-attribution demo
              purposes only. No real RBAC backs this; it's a labeling
              attribute that shows up on the /dashboard distribution chart. */}
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs" style={{ color: "var(--ink-4)" }}>
              department:
            </span>
            <select
              value={selectedDepartment}
              onChange={(e) => setSelectedDepartment(e.target.value)}
              disabled={loading}
              className="font-mono text-xs px-2 py-1 rounded-lg border transition-colors disabled:opacity-40"
              style={{
                borderColor: "var(--rule)",
                background: "var(--paper)",
                color: "var(--ink-2)",
              }}
            >
              <option value="">unspecified</option>
              {DEPARTMENTS.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          {/* Force cloud toggle — only meaningful in production mode.
              In demo mode the router already sends every request to
              OpenRouter (skips Ollama), so the toggle is a no-op and
              would only mislead. */}
          {demoMode === false && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => setForceCloud((v) => !v)}
                className="flex items-center gap-2 font-mono text-xs transition-colors"
                style={{ color: forceCloud ? "var(--coral)" : "var(--ink-4)" }}
              >
                <span
                  className="inline-block w-3 h-3 rounded border-2 transition-colors"
                  style={{
                    borderColor: forceCloud ? "var(--coral)" : "var(--rule)",
                    background: forceCloud ? "var(--coral)" : "transparent",
                  }}
                />
                Force cloud
              </button>
              <span className="font-mono text-xs" style={{ color: "var(--ink-4)" }}>
                (bypass Ollama → OpenRouter)
              </span>
            </div>
          )}

          {/* Streaming toggle — renders the answer token-by-token via SSE
              instead of waiting for the full completion. */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setStreamingEnabled((v) => !v)}
              className="flex items-center gap-2 font-mono text-xs transition-colors"
              style={{ color: streamingEnabled ? "var(--accent)" : "var(--ink-4)" }}
            >
              <span
                className="inline-block w-3 h-3 rounded border-2 transition-colors"
                style={{
                  borderColor: streamingEnabled ? "var(--accent)" : "var(--rule)",
                  background: streamingEnabled ? "var(--accent)" : "transparent",
                }}
              />
              Stream response
            </button>
          </div>
        </div>

        {/* Text area + send */}
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the documents…"
            rows={2}
            disabled={loading}
            className="flex-1 resize-none rounded-lg border px-3 py-2 text-sm focus:outline-none focus:ring-2 transition-all"
            style={{
              borderColor: "var(--rule)",
              background: "var(--paper)",
              color: "var(--ink)",
              fontFamily: "Instrument Sans, sans-serif",
            }}
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={loading || !input.trim()}
            className="rounded-lg px-4 py-2 text-sm font-medium transition-all disabled:opacity-40"
            style={{
              background: "var(--accent)",
              color: "#fff",
              height: "fit-content",
            }}
          >
            {loading ? "…" : "Send"}
          </button>
        </div>
        <p className="font-mono text-xs" style={{ color: "var(--ink-4)" }}>
          Press Enter to send · Shift+Enter for new line
        </p>
      </div>
    </div>
  );
}
