"use client";

import { useCallback, useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { getDashboardStats, getRecentTraces } from "@/lib/api";
import type { InferenceProvider, TelemetryStats, TraceSummary } from "@/lib/api";
import { PROVIDER_CONFIG } from "@/lib/providerStyles";

const POLL_INTERVAL_MS = 5000;
const WINDOW_SECONDS = 3600;

const DISTRIBUTION_PALETTE = ["var(--accent)", "var(--teal)", "var(--amber)", "var(--coral)"];

function formatMs(ms: number | null): string {
  if (ms === null) return "—";
  return ms < 10 ? "<10 ms" : `${Math.round(ms)} ms`;
}

function formatPct(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

function timeAgo(epochSeconds: number): string {
  const deltaSeconds = Math.max(0, Date.now() / 1000 - epochSeconds);
  if (deltaSeconds < 60) return `${Math.round(deltaSeconds)}s ago`;
  if (deltaSeconds < 3600) return `${Math.round(deltaSeconds / 60)}m ago`;
  return `${Math.round(deltaSeconds / 3600)}h ago`;
}

function CardShell({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string;
  title: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-xl border p-5" style={{ borderColor: "var(--rule)", background: "#fff" }}>
      <div className="mb-4 pb-3 border-b" style={{ borderColor: "var(--rule)" }}>
        <div className="font-mono text-xs mb-1" style={{ color: "var(--ink-4)", letterSpacing: "0.08em" }}>
          {eyebrow}
        </div>
        <h2 className="text-base font-semibold" style={{ color: "var(--ink)" }}>
          {title}
        </h2>
        {description && (
          <p className="text-xs mt-1" style={{ color: "var(--ink-3)", lineHeight: 1.6 }}>
            {description}
          </p>
        )}
      </div>
      {children}
    </div>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3" style={{ borderColor: "var(--rule)", background: "var(--paper)" }}>
      <div className="font-mono text-xs mb-1" style={{ color: "var(--ink-4)", letterSpacing: "0.06em" }}>
        {label}
      </div>
      <div className="font-mono text-lg font-medium" style={{ color: "var(--ink)" }}>
        {value}
      </div>
    </div>
  );
}

function DistributionRow({
  label,
  count,
  total,
  color,
}: {
  label: string;
  count: number;
  total: number;
  color: string;
}) {
  const pct = total > 0 ? count / total : 0;
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-xs w-28 flex-shrink-0 truncate" style={{ color: "var(--ink-3)" }}>
        {label}
      </span>
      <div className="flex-1 h-2 rounded-full overflow-hidden" style={{ background: "var(--rule-light)" }}>
        <div
          className="h-full rounded-full transition-all"
          style={{ width: `${Math.max(pct * 100, count > 0 ? 3 : 0)}%`, background: color }}
        />
      </div>
      <span className="font-mono text-xs w-16 text-right flex-shrink-0" style={{ color: "var(--ink-4)" }}>
        {count} ({formatPct(pct)})
      </span>
    </div>
  );
}

function PieChart({
  entries,
  palette,
}: {
  entries: [string, number][];
  palette: string[];
}) {
  const total = entries.reduce((sum, [, c]) => sum + c, 0);

  let cumulativePct = 0;
  const stops = entries.map(([, count], i) => {
    const startPct = cumulativePct;
    const slicePct = total > 0 ? (count / total) * 100 : 0;
    cumulativePct += slicePct;
    return `${palette[i % palette.length]} ${startPct}% ${cumulativePct}%`;
  });
  const gradient = stops.length > 0 ? `conic-gradient(${stops.join(", ")})` : "var(--rule-light)";

  return (
    <div className="flex items-center gap-5">
      <div
        className="rounded-full flex-shrink-0"
        style={{ width: 120, height: 120, background: gradient }}
      />
      <div className="flex-1 space-y-2 min-w-0">
        {entries.length === 0 && (
          <p className="text-xs" style={{ color: "var(--ink-4)" }}>
            No requests recorded yet.
          </p>
        )}
        {entries.map(([label, count], i) => (
          <div key={label} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0"
              style={{ background: palette[i % palette.length] }}
            />
            <span className="font-mono text-xs flex-1 truncate" style={{ color: "var(--ink-3)" }}>
              {label}
            </span>
            <span className="font-mono text-xs flex-shrink-0" style={{ color: "var(--ink-4)" }}>
              {count} ({formatPct(total > 0 ? count / total : 0)})
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ProviderBadge({ provider }: { provider: InferenceProvider | null }) {
  if (!provider) {
    return (
      <span className="font-mono text-xs" style={{ color: "var(--ink-4)" }}>
        —
      </span>
    );
  }
  const cfg = PROVIDER_CONFIG[provider];
  return (
    <span
      className="font-mono text-xs font-medium px-2 py-0.5 rounded-full border inline-flex items-center gap-1.5"
      style={{ background: cfg.bg, color: cfg.color, borderColor: cfg.border }}
    >
      <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: cfg.dot }} />
      {cfg.label}
    </span>
  );
}

export default function DashboardPage() {
  const { status } = useSession();
  const router = useRouter();
  const [stats, setStats] = useState<TelemetryStats | null>(null);
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (status === "unauthenticated") router.push("/auth/signin");
  }, [status, router]);

  const refresh = useCallback(async () => {
    try {
      const [statsResult, tracesResult] = await Promise.all([
        getDashboardStats(WINDOW_SECONDS),
        getRecentTraces(50),
      ]);
      setStats(statsResult);
      setTraces(tracesResult);
      setLoadError(null);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : "Failed to load telemetry.");
    }
  }, []);

  useEffect(() => {
    if (status !== "authenticated") return;
    refresh();
    const id = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [status, refresh]);

  if (status === "loading" || status === "unauthenticated") {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--paper)", color: "var(--ink)" }}>
        <svg className="w-6 h-6 animate-spin" viewBox="0 0 24 24" fill="none" style={{ color: "var(--accent)" }}>
          <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" opacity="0.25" />
          <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        </svg>
      </div>
    );
  }

  const providerEntries = Object.entries(stats?.provider_distribution ?? {});
  const modelEntries = Object.entries(stats?.model_distribution ?? {});
  const departmentEntries = Object.entries(stats?.department_distribution ?? {});
  const providerTotal = providerEntries.reduce((sum, [, c]) => sum + c, 0);
  const modelTotal = modelEntries.reduce((sum, [, c]) => sum + c, 0);

  return (
    <div className="min-h-screen" style={{ background: "var(--paper)", color: "var(--ink)" }}>
      <header className="border-b px-6 py-4" style={{ borderColor: "var(--rule)" }}>
        <div className="max-w-7xl mx-auto flex items-center justify-between flex-wrap gap-3">
          <div>
            <div
              className="font-mono text-xs font-medium mb-1 flex items-center gap-2"
              style={{ color: "var(--accent-mid)", letterSpacing: "0.1em" }}
            >
              <span className="inline-block w-5 h-px" style={{ background: "var(--accent-mid)" }} />
              OBSERVABILITY
            </div>
            <h1 className="font-serif text-2xl leading-tight" style={{ color: "var(--ink)", letterSpacing: "-0.01em" }}>
              Request{" "}
              <em className="italic" style={{ color: "var(--accent)" }}>
                Telemetry
              </em>
            </h1>
          </div>
          <a
            href="/"
            className="font-mono text-xs px-3 py-1 rounded-full border transition-colors hover:opacity-80"
            style={{ borderColor: "var(--rule)", background: "var(--rule-light)", color: "var(--ink-3)" }}
          >
            ← Back to demo
          </a>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {loadError && (
          <div
            className="rounded-lg border px-4 py-3 text-sm"
            style={{ borderColor: "#f5c4b3", background: "var(--coral-light)", color: "var(--coral)" }}
          >
            {loadError}
          </div>
        )}

        <CardShell
          eyebrow="LAST 60 MINUTES"
          title="Summary"
          description="Aggregated from traces recorded across /query/ and /query/stream — recomputed on every poll, not incrementally counted."
        >
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
            <StatTile label="requests" value={String(stats?.total_requests ?? 0)} />
            <StatTile label="errors" value={String(stats?.error_count ?? 0)} />
            <StatTile label="cache hit rate" value={formatPct(stats?.cache_hit_rate ?? 0)} />
            <StatTile label="rag ratio" value={formatPct(stats?.rag_ratio ?? 0)} />
            <StatTile label="p50 latency" value={formatMs(stats?.latency_p50_ms ?? null)} />
            <StatTile label="p95 latency" value={formatMs(stats?.latency_p95_ms ?? null)} />
            <StatTile label="avg ttft" value={formatMs(stats?.avg_ttft_ms ?? null)} />
          </div>
        </CardShell>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <CardShell eyebrow="ROUTING" title="Provider distribution">
            <div className="space-y-3">
              {providerEntries.length === 0 && (
                <p className="text-xs" style={{ color: "var(--ink-4)" }}>
                  No requests recorded yet.
                </p>
              )}
              {providerEntries.map(([provider, count]) => (
                <DistributionRow
                  key={provider}
                  label={provider}
                  count={count}
                  total={providerTotal}
                  color={PROVIDER_CONFIG[provider as InferenceProvider]?.color ?? "var(--accent)"}
                />
              ))}
            </div>
          </CardShell>

          <CardShell eyebrow="ROUTING" title="Model distribution">
            <div className="space-y-3">
              {modelEntries.length === 0 && (
                <p className="text-xs" style={{ color: "var(--ink-4)" }}>
                  No requests recorded yet.
                </p>
              )}
              {modelEntries.map(([model, count], i) => (
                <DistributionRow
                  key={model}
                  label={model}
                  count={count}
                  total={modelTotal}
                  color={DISTRIBUTION_PALETTE[i % DISTRIBUTION_PALETTE.length]}
                />
              ))}
            </div>
          </CardShell>

          <CardShell
            eyebrow="SIMULATED USERS"
            title="Department distribution"
            description="Self-reported in chat, or randomly assigned per virtual user by the load simulator. Not real RBAC."
          >
            <PieChart entries={departmentEntries} palette={DISTRIBUTION_PALETTE} />
          </CardShell>
        </div>

        <CardShell eyebrow="RECENT ACTIVITY" title="Request trace" description="Newest first, polling every 5s.">
          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ color: "var(--ink-4)" }}>
                  {["time", "route", "query", "provider", "cached", "latency", "ttft", "model"].map((h) => (
                    <th key={h} className="font-mono text-left font-medium pb-2 pr-4" style={{ letterSpacing: "0.06em" }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {traces.length === 0 && (
                  <tr>
                    <td colSpan={8} className="py-4 text-center" style={{ color: "var(--ink-4)" }}>
                      No traces yet — send a query to populate this table.
                    </td>
                  </tr>
                )}
                {traces.map((t) => (
                  <tr key={t.trace_id} className="border-t" style={{ borderColor: "var(--rule-light)" }}>
                    <td className="py-2 pr-4 font-mono" style={{ color: "var(--ink-4)" }}>
                      {timeAgo(t.started_at)}
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: "var(--ink-3)" }}>
                      {t.route}
                    </td>
                    <td className="py-2 pr-4 max-w-xs truncate" style={{ color: "var(--ink-2)" }} title={t.query_preview}>
                      {t.error ? (
                        <span style={{ color: "var(--coral)" }}>⚠ {t.error}</span>
                      ) : (
                        t.query_preview
                      )}
                    </td>
                    <td className="py-2 pr-4">
                      <ProviderBadge provider={t.provider} />
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: "var(--ink-3)" }}>
                      {t.cached ? "yes" : "no"}
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: "var(--ink-3)" }}>
                      {formatMs(t.total_latency_ms)}
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: "var(--ink-3)" }}>
                      {formatMs(t.ttft_ms)}
                    </td>
                    <td className="py-2 pr-4 font-mono" style={{ color: "var(--ink-3)" }}>
                      {t.model_alias ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardShell>
      </main>
    </div>
  );
}
