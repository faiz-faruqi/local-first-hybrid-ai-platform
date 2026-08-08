/**
 * Typed API client for the FastAPI backend.
 * Uses NEXT_PUBLIC_API_URL environment variable.
 */

const rawApiBase = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_BASE = rawApiBase.endsWith("/") ? rawApiBase.slice(0, -1) : rawApiBase;

export type InferenceProvider = "local" | "cloud" | "cache";

export interface SourceDocument {
  chunk_id: string;
  document_name: string;
  content_preview: string;
  relevance_score: number;
}

// ── Phase 2: Query classification ───────────────────────────────────────────

export type Complexity = "low" | "medium" | "high";
export type Domain = "general" | "finance" | "healthcare" | "legal" | "coding" | "enterprise";
export type Sensitivity = "public" | "internal" | "confidential";
export type ContextSize = "small" | "medium" | "large";
export type LatencyTier = "interactive" | "batch";

export interface QueryProfile {
  complexity: Complexity;
  domain: Domain;
  sensitivity: Sensitivity;
  context_size: ContextSize;
  rag_needed: boolean;
  latency_tier: LatencyTier;
  confidence: number;
  token_estimate: number;
  signals: string[];
}

// ── Phase 3: Routing decision ───────────────────────────────────────────────

export interface RoutingDecision {
  selected_model: string;
  selected_tier: string;
  reason: string;
  fallback_chain: string[];
  estimated_cost: number;
  budget_remaining: number | null;
  rules_matched: string[];
}

// ── Phase 1: Model catalog ──────────────────────────────────────────────────

export interface ModelInfo {
  alias: string;
  vendor: string;
  model_id: string;
  tier: string;
  context_window: number;
  cost_per_1k_input: number;
  cost_per_1k_output: number;
  latency_tier: string;
  is_local: boolean;
  description: string;
}

// ── Request / Response ──────────────────────────────────────────────────────

export interface QueryRequest {
  query: string;
  top_k?: number;
  force_cloud?: boolean;
  model?: string | null;
  department?: string | null;
}

export interface QueryResponse {
  answer: string;
  provider: InferenceProvider;
  cached: boolean;
  sources: SourceDocument[];
  latency_ms: number;
  model_alias?: string | null;
  classification?: QueryProfile | null;
  routing_decision?: RoutingDecision | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  demo_mode: boolean;
  vector_store: string;
  components: Record<string, string>;
}

// ── Telemetry / observability ────────────────────────────────────────────────

export interface TelemetryStats {
  window_seconds: number;
  total_requests: number;
  error_count: number;
  cache_hit_rate: number;
  rag_ratio: number;
  latency_p50_ms: number | null;
  latency_p95_ms: number | null;
  latency_p99_ms: number | null;
  avg_ttft_ms: number | null;
  provider_distribution: Record<string, number>;
  model_distribution: Record<string, number>;
  department_distribution: Record<string, number>;
}

export interface TraceSummary {
  trace_id: string;
  route: string;
  query_preview: string;
  started_at: number;
  total_latency_ms: number;
  ttft_ms: number | null;
  provider: InferenceProvider | null;
  cached: boolean;
  model_alias?: string | null;
  rag_needed?: boolean | null;
  error?: string | null;
}

export async function queryDocuments(req: QueryRequest): Promise<QueryResponse> {
  const res = await fetch(`${API_BASE}/query/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: HTTP ${res.status}`);
  return res.json();
}

export async function getModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models/`);
  if (!res.ok) throw new Error(`Failed to fetch models: HTTP ${res.status}`);
  return res.json();
}

export async function getDashboardStats(windowSeconds = 3600): Promise<TelemetryStats> {
  const res = await fetch(`${API_BASE}/telemetry/stats?window_seconds=${windowSeconds}`);
  if (!res.ok) throw new Error(`Failed to fetch telemetry stats: HTTP ${res.status}`);
  return res.json();
}

export async function getRecentTraces(limit = 50): Promise<TraceSummary[]> {
  const res = await fetch(`${API_BASE}/telemetry/traces?limit=${limit}`);
  if (!res.ok) throw new Error(`Failed to fetch traces: HTTP ${res.status}`);
  return res.json();
}

// ── Streaming query (SSE) ────────────────────────────────────────────────────
//
// Deliberately not the native EventSource API — EventSource only supports
// GET requests with no custom body, and this endpoint is a POST carrying
// the QueryRequest JSON payload, so we parse the SSE frame format by hand
// over a plain fetch + ReadableStream instead.

export interface StreamDonePayload {
  provider: InferenceProvider;
  cached: boolean;
  latency_ms: number;
  model_alias?: string | null;
  sources: SourceDocument[];
  classification?: QueryProfile | null;
  routing_decision?: RoutingDecision | null;
}

export async function streamQuery(
  req: QueryRequest,
  onDelta: (text: string) => void,
  onDone: (payload: StreamDonePayload) => void,
  onError: (message: string) => void
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch (err) {
    onError(err instanceof Error ? err.message : "Network error");
    return;
  }

  if (!res.ok || !res.body) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    onError(err.detail ?? `HTTP ${res.status}`);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      if (!rawEvent.startsWith("data:")) continue;
      const payload = JSON.parse(rawEvent.slice("data:".length).trim());

      if (payload.event === "done") {
        onDone(payload as StreamDonePayload);
      } else if (payload.event === "error") {
        onError(payload.message ?? "Stream error");
      } else if (typeof payload.delta === "string") {
        onDelta(payload.delta);
      }
    }
  }
}
