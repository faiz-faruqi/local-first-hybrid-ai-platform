"use client";

import type { InferenceProvider, QueryProfile, RoutingDecision } from "@/lib/api";
import { PROVIDER_CONFIG } from "@/lib/providerStyles";

interface ResponseMetaProps {
  provider: InferenceProvider;
  cached: boolean;
  latencyMs: number;
  modelAlias?: string | null;
  classification?: QueryProfile | null;
  routingDecision?: RoutingDecision | null;
}

const TIER_STYLES: Record<string, { bg: string; color: string; border: string }> = {
  local: { bg: "#e6f1fb", color: "#185fa5", border: "#b5d4f4" },
  cheap: { bg: "#e1f5ee", color: "#0f6e56", border: "#9fe1cb" },
  standard: { bg: "#fef9e7", color: "#8a6d1b", border: "#f5e9b3" },
  premium: { bg: "#f3e8fd", color: "#6b21a8", border: "#d4b3f5" },
};

function TierBadge({ tier }: { tier: string }) {
  const style = TIER_STYLES[tier] ?? TIER_STYLES.standard;
  return (
    <span
      className="font-mono text-xs font-medium px-2 py-0.5 rounded border uppercase"
      style={style}
    >
      {tier}
    </span>
  );
}

function ClassificationPill({ label, value }: { label: string; value: string }) {
  return (
    <span
      className="font-mono text-xs px-2 py-0.5 rounded border"
      style={{ background: "var(--paper)", color: "var(--ink-3)", borderColor: "var(--rule)" }}
    >
      {label}: <strong style={{ color: "var(--ink-2)" }}>{value}</strong>
    </span>
  );
}

export default function ResponseMeta({
  provider,
  cached,
  latencyMs,
  modelAlias,
  classification,
  routingDecision,
}: ResponseMetaProps) {
  const cfg = PROVIDER_CONFIG[provider];

  return (
    <div className="mt-2 space-y-2">
      {/* Primary metadata row */}
      <div className="flex items-center gap-3 flex-wrap">
        {/* Provider badge */}
        <span
          className="font-mono text-xs font-medium px-3 py-1 rounded-full border flex items-center gap-1.5"
          style={{ background: cfg.bg, color: cfg.color, borderColor: cfg.border }}
        >
          <span
            className="inline-block w-1.5 h-1.5 rounded-full"
            style={{ background: cfg.dot }}
          />
          {cfg.label}
        </span>

        {/* Model alias (Phase 1) */}
        {modelAlias && (
          <span
            className="font-mono text-xs px-2 py-0.5 rounded border"
            style={{ background: "var(--paper)", color: "var(--ink-2)", borderColor: "var(--rule)" }}
          >
            model: <strong>{modelAlias}</strong>
          </span>
        )}

        {/* Routing tier badge (Phase 3) */}
        {routingDecision && (
          <TierBadge tier={routingDecision.selected_tier} />
        )}

        {/* Cache pill */}
        {cached && (
          <span
            className="font-mono text-xs px-2 py-0.5 rounded border"
            style={{ background: "#e1f5ee", color: "#0f6e56", borderColor: "#9fe1cb" }}
          >
            cache hit
          </span>
        )}

        {/* Latency */}
        <span
          className="font-mono text-xs"
          style={{ color: "var(--ink-4)" }}
        >
          {latencyMs < 10 ? `<10 ms` : `${Math.round(latencyMs)} ms`}
        </span>

        {/* Estimated cost (Phase 3) */}
        {routingDecision && routingDecision.estimated_cost > 0 && (
          <span
            className="font-mono text-xs"
            style={{ color: "var(--ink-4)" }}
          >
            est. ${routingDecision.estimated_cost.toFixed(6)}
          </span>
        )}
      </div>

      {/* Classification dimensions (Phase 2) */}
      {classification && (
        <div className="flex items-center gap-2 flex-wrap">
          <span
            className="font-mono text-xs font-medium"
            style={{ color: "var(--ink-4)" }}
          >
            classification:
          </span>
          <ClassificationPill label="complexity" value={classification.complexity} />
          <ClassificationPill label="domain" value={classification.domain} />
          <ClassificationPill label="sensitivity" value={classification.sensitivity} />
          <ClassificationPill label="context" value={classification.context_size} />
          <ClassificationPill label="rag" value={classification.rag_needed ? "yes" : "no"} />
          <ClassificationPill label="latency" value={classification.latency_tier} />
          <ClassificationPill
            label="confidence"
            value={`${(classification.confidence * 100).toFixed(0)}%`}
          />
        </div>
      )}

      {/* Routing decision reason (Phase 3) */}
      {routingDecision && (
        <div className="flex items-start gap-2 flex-wrap">
          <span
            className="font-mono text-xs font-medium"
            style={{ color: "var(--ink-4)" }}
          >
            routing:
          </span>
          <span
            className="font-mono text-xs px-2 py-0.5 rounded border"
            style={{ background: "var(--paper)", color: "var(--ink-2)", borderColor: "var(--rule)" }}
          >
            {routingDecision.reason}
          </span>
          {routingDecision.fallback_chain.length > 0 && (
            <span
              className="font-mono text-xs"
              style={{ color: "var(--ink-4)" }}
            >
              fallback: {routingDecision.fallback_chain.join(" → ")}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
