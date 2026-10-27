import { useEffect, useState } from "react";

export type ContributionStatus = "uploading" | "queued" | "processing" | "done" | "rejected" | "failed";

export type Scene = {
  id: number;
  slug: string;
  name: string;
  created_at: string;
  updated_at: string;
  head_version_id: number | null;
  counts?: Partial<Record<ContributionStatus, number>>;
};

export type TimelineEntry = {
  name: string;
  status: "done" | "active" | "pending";
  seconds: number | null;
  estimate: number;
  remaining: number | null;
  done?: number;
  total?: number;
  substeps?: TimelineEntry[];
};

export type Contribution = {
  id: number;
  scene_id: number;
  seq: number;
  filename: string;
  status: ContributionStatus;
  stage: string | null;
  stage_progress: number;
  progress: number;
  eta_seconds: number | null;
  preview_url: string | null;
  preview_step: number | null;
  detail: string | null;
  timeline: TimelineEntry[] | null;
  error: string | null;
  metrics: Record<string, unknown>;
  version_id: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
};

export type SplatKind = "" | "points" | "draft" | "quick" | "full" | "uniform";
export type TrainQuality = "quick" | "full" | "uniform";

export type Version = {
  id: number;
  scene_id: number;
  number: number;
  parent_id: number | null;
  contribution_id: number | null;
  contribution_ids: number[];
  splat_url: string;
  dense_url: string;
  layers_url: string;
  metrics: Record<string, unknown>;
  splat_kind: SplatKind;
  /** Every trained kind's LoD URL (splat_kind is the latest). */
  splats: Partial<Record<SplatKind, string>>;
  train_status: "queued" | "training" | "cancelling" | "failed" | null;
  train_quality: TrainQuality | null;
  train_auto: boolean;
  train_progress: number;
  train_eta: number | null;
  train_detail: string | null;
  train_timeline: TimelineEntry[] | null;
  train_error: string | null;
  preview_url: string | null;
  preview_step: number | null;
  created_at: string;
};

export type SceneState = {
  scene: Scene;
  head: Version | null;
  versions: Version[];
  contributions: Contribution[];
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  listScenes: () => fetch("/api/scenes").then((r) => json<Scene[]>(r)),
  createScene: (name: string) =>
    fetch("/api/scenes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }).then((r) => json<Scene>(r)),
  setHead: (slug: string, versionId: number) =>
    fetch(`/api/scenes/${slug}/head`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version_id: versionId }),
    }).then((r) => json<SceneState>(r)),
  train: (slug: string, versionId: number, quality: TrainQuality) =>
    fetch(`/api/scenes/${slug}/versions/${versionId}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quality }),
    }).then((r) => json<SceneState>(r)),
  cancelTraining: (slug: string, versionId: number) =>
    fetch(`/api/scenes/${slug}/versions/${versionId}/train`, { method: "DELETE" }).then((r) => json<SceneState>(r)),
  /** XHR (not fetch) so we get upload progress events. */
  uploadVideo: (slug: string, file: File, onProgress: (fraction: number) => void) =>
    new Promise<Contribution>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/scenes/${slug}/videos`);
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () =>
        xhr.status < 300 ? resolve(JSON.parse(xhr.responseText)) : reject(new Error(xhr.responseText));
      xhr.onerror = () => reject(new Error("upload failed"));
      const form = new FormData();
      form.append("file", file);
      xhr.send(form);
    }),
};

/** Live scene state over SSE; the server pushes the full state whenever anything changes. */
export function useSceneState(slug: string | null): { state: SceneState | null; connected: boolean } {
  const [state, setState] = useState<SceneState | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    setState(null);
    if (!slug) return;
    const source = new EventSource(`/api/scenes/${slug}/events`);
    source.addEventListener("state", (e) => setState(JSON.parse((e as MessageEvent).data)));
    source.onopen = () => setConnected(true);
    source.onerror = () => setConnected(false);
    return () => source.close();
  }, [slug]);

  return { state, connected };
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "–";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/** Same palette as worker/pipeline/export.py CONTRIBUTION_COLORS (by seq - 1). */
export const CONTRIBUTION_COLORS = ["#468cff", "#ff9628", "#c850dc", "#3cc8c8", "#e6465a"];
export const SHARED_COLOR = "#50dc64";
export const contributionColor = (seq: number) => CONTRIBUTION_COLORS[(seq - 1) % CONTRIBUTION_COLORS.length];

/** Viewer order of trained splat kinds. */
export const SPLAT_KINDS: SplatKind[] = ["full", "quick", "uniform"];

export const SPLAT_KIND_LABELS: Record<SplatKind, string> = {
  "": "no splats",
  points: "points preview",
  draft: "draft (new areas as points)",
  quick: "quick splats",
  full: "full-quality splats",
  uniform: "uniform splats",
};
