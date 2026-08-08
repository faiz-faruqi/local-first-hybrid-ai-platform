/**
 * Shared provider badge styling.
 *
 * Extracted from ResponseMeta so the dashboard's trace table can render
 * the same LOCAL/CLOUD/CACHED badges without duplicating the color map.
 */

import type { InferenceProvider } from "./api";

export const PROVIDER_CONFIG: Record<
  InferenceProvider,
  { label: string; bg: string; color: string; border: string; dot: string }
> = {
  local: {
    label: "LOCAL",
    bg: "#e6f1fb",
    color: "#185fa5",
    border: "#b5d4f4",
    dot: "#185fa5",
  },
  cloud: {
    label: "CLOUD",
    bg: "#faece7",
    color: "#993c1d",
    border: "#f5c4b3",
    dot: "#993c1d",
  },
  cache: {
    label: "CACHED",
    bg: "#e1f5ee",
    color: "#0f6e56",
    border: "#9fe1cb",
    dot: "#0f6e56",
  },
};
