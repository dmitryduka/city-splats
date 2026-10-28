import { useRef, useState } from "react";
import {
  api,
  contributionColor,
  formatDuration,
  SPLAT_KIND_LABELS,
  type Contribution,
  type SceneState,
  type TimelineEntry,
  type TrainQuality,
  type Version,
} from "../api";

const STAGE_LABELS: Record<string, string> = {
  ingest: "Extracting frames",
  sfm: "Reconstructing (SfM)",
  depth: "Depth maps",
  register: "Fitting into scene",
  layers: "Exporting point layers",
  dense: "Dense point cloud",
  frames: "Preparing training frames",
  init: "Dense init",
  train: "Training splats",
  cleanup: "Removing floaters",
  lod: "Building LoD for streaming",
  published: "Published",
};

const STEP_LABELS: Record<string, string> = {
  extract: "Features",
  match_sequential: "Match within video",
  match_weak_links: "Re-match weak links",
  match_retrieval: "Match across videos",
  map: "Mapping",
  submodel_merge: "Submodel fallback",
  gate: "Consistency check",
  cameras: "One camera per video",
  pick: "Pick extra frames",
  decode: "Decode full-res frames",
  pose: "Pose extra frames",
  sample: "Read dense tiles",
};

function TimelineRow({ entry, label, nested }: { entry: TimelineEntry; label: string; nested?: boolean }) {
  const icon = entry.status === "done" ? "✓" : entry.status === "active" ? "▶" : "·";
  const counts = entry.total ? ` ${entry.done ?? 0}/${entry.total}` : "";
  const time =
    entry.status === "done"
      ? entry.seconds != null
        ? formatDuration(entry.seconds)
        : ""
      : entry.status === "active"
        ? entry.remaining != null
          ? `${formatDuration(entry.seconds)} · ~${formatDuration(entry.remaining)} left`
          : formatDuration(entry.seconds)
        : entry.estimate
          ? `~${formatDuration(entry.estimate)}`
          : "";
  return (
    <div className={`tl-row ${entry.status} ${nested ? "nested" : ""}`}>
      <span className="tl-icon">{icon}</span>
      <span className="tl-label">
        {label}
        {counts}
      </span>
      <span className="tl-time">{time}</span>
    </div>
  );
}

function Timeline({ timeline }: { timeline: TimelineEntry[] }) {
  return (
    <div className="timeline">
      {timeline.map((stage) => (
        <div key={stage.name}>
          <TimelineRow entry={stage} label={STAGE_LABELS[stage.name] ?? stage.name} />
          {stage.status !== "pending" &&
            stage.substeps?.map((sub) => (
              <TimelineRow key={sub.name} entry={sub} label={STEP_LABELS[sub.name] ?? sub.name} nested />
            ))}
        </div>
      ))}
    </div>
  );
}

type Props = {
  state: SceneState;
  viewingVersionId: number | null;
  onViewVersion: (versionId: number | null) => void;
  onError: (message: string) => void;
};

function Upload({ slug, onError }: { slug: string; onError: (m: string) => void }) {
  const [uploads, setUploads] = useState<{ name: string; progress: number }[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const send = async (files: FileList | null) => {
    for (const file of Array.from(files ?? [])) {
      setUploads((u) => [...u, { name: file.name, progress: 0 }]);
      try {
        await api.uploadVideo(slug, file, (progress) =>
          setUploads((u) => u.map((x) => (x.name === file.name ? { ...x, progress } : x))),
        );
      } catch (err) {
        onError(`Upload of ${file.name} failed: ${(err as Error).message}`);
      } finally {
        setUploads((u) => u.filter((x) => x.name !== file.name));
      }
    }
  };

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        send(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        multiple
        hidden
        onChange={(e) => {
          send(e.target.files);
          e.target.value = "";
        }}
      />
      {uploads.length === 0 ? (
        <span>Drop videos here or click to upload</span>
      ) : (
        uploads.map((u) => (
          <div key={u.name} className="upload-row">
            <span>{u.name}</span>
            <div className="bar">
              <div style={{ width: `${u.progress * 100}%` }} />
            </div>
          </div>
        ))
      )}
    </div>
  );
}

function ContributionCard({ c }: { c: Contribution }) {
  const m = c.metrics as Record<string, number | string | boolean | undefined>;
  // A queued video whose own stages (frames, 3D model, depth) run ahead of the queue shows
  // as processing; only fitting into the scene waits for the videos before it.
  const ahead = c.status === "queued" && (c.timeline ?? []).some((s) => s.status !== "pending");
  const active = c.status === "processing" || ahead;
  const running = (c.timeline ?? []).filter((s) => s.status === "active");
  const stage = ahead
    ? running.map((s) => STAGE_LABELS[s.name] ?? s.name).join(" + ") || "Ready, fitting waits for earlier videos"
    : (STAGE_LABELS[c.stage ?? ""] ?? c.stage ?? "starting");
  const [showTimings, setShowTimings] = useState(false);
  return (
    <li className={`contribution ${active ? "processing" : c.status}`}>
      <div className="row">
        <span className="dot" style={{ background: contributionColor(c.seq) }} />
        <span className="title">
          #{c.seq} {c.filename}
        </span>
        <span className={`badge ${active ? "processing" : c.status}`}>{active ? "processing" : c.status}</span>
      </div>
      {active && (
        <>
          <div className="row muted">
            <span>{stage}</span>
            <span className="eta">{ahead ? (running.length ? "ahead of queue" : "") : `ETA ${formatDuration(c.eta_seconds)}`}</span>
          </div>
          <div className="bar">
            <div style={{ width: `${c.progress * 100}%` }} />
          </div>
          {c.detail && (!ahead || running.length > 0) && <div className="muted small">{c.detail}</div>}
          {c.timeline && c.timeline.length > 0 && <Timeline timeline={c.timeline} />}
        </>
      )}
      {!active && c.timeline && c.timeline.length > 0 && (
        <>
          <button className="link small" onClick={() => setShowTimings((v) => !v)}>
            {showTimings ? "hide timings" : "timings"}
          </button>
          {showTimings && <Timeline timeline={c.timeline} />}
        </>
      )}
      {c.status === "queued" && !ahead && <div className="muted small">Waiting to start</div>}
      {(m.keyframes != null || m.registered != null) && (
        <div className="muted small">
          {[
            m.keyframes != null && `${m.keyframes} keyframes`,
            m.registered != null &&
              `${m.registered} registered` +
                (m.registration_ratio != null ? ` (${Math.round(Number(m.registration_ratio) * 100)}%)` : ""),
            m.points != null && `${Number(m.points).toLocaleString()} pts`,
            m.reproj_error_px != null && `${m.reproj_error_px}px`,
          ]
            .filter(Boolean)
            .join(" · ")}
        </div>
      )}
      {typeof m.gate === "string" && c.status === "done" && (
        <div className="muted small">
          fit: {String(m.strategy ?? "")} · {m.gate}
        </div>
      )}
      {c.error && <div className="error small">{c.error.split("\n")[0]}</div>}
    </li>
  );
}

const TRAIN_OPTIONS: { quality: TrainQuality; label: string; title: string }[] = [
  { quality: "quick", label: "Train splats", title: "Quick: 3000 steps at 1024 px, a few minutes. Live preview while training." },
  {
    quality: "uniform",
    label: "Uniform splats",
    title: "Round splats with one color each (no view-dependent shading), sized by distance from the cameras; size adjustable in the viewer ([ ]).",
  },
  { quality: "full", label: "Full quality", title: "7000 steps at 1600 px; best quality, ~4x longer" },
];

type SkyReport = { total: number; sky: number; above_roofs: number };
type BlockReport = { frames: number; train_frames: number; eval?: { psnr: number } | null; kept?: number; sky?: SkyReport | null };
type FramesReport = {
  keyframes: number;
  full_resolution: number;
  extra_candidates: number;
  extra_posed: number;
  extra_held_out: number;
  reposed: number;
  dropped: number;
};
type TrainReport = { frames?: FramesReport | null; blocks?: Record<string, BlockReport>; steps?: number; max_resolution?: number };

/** What block training found: frames prepared (worker.pipeline.splat.frames) and each
 * finished block's held-out PSNR; live while training, then from the splat metrics. */
function TrainingReport({ report, skyRemoved }: { report: TrainReport; skyRemoved?: Record<string, SkyReport> }) {
  const f = report.frames;
  const blocks = Object.entries(report.blocks ?? {});
  // Sky removal (worker.pipeline.sky): by the worker per block, or by scripts.remove_sky.
  const sky = blocks
    .map(([key, b]) => [key, b.sky ?? skyRemoved?.[key]] as const)
    .filter((e): e is readonly [string, SkyReport] => e[1] != null);
  const psnrs = blocks.map(([, b]) => b.eval?.psnr).filter((p): p is number => p != null);
  const mean = psnrs.length ? psnrs.reduce((a, b) => a + b, 0) / psnrs.length : null;
  return (
    <div className="muted small report">
      {f && (
        <div>
          Frames: {f.keyframes} keyframes{f.full_resolution ? " at full resolution" : ""}
          {f.extra_candidates > 0 &&
            ` · +${f.extra_posed} extra (${f.extra_candidates} picked, ${f.extra_held_out} skipped next to held-out frames)`}
          {(f.reposed > 0 || f.dropped > 0) && ` · ${f.reposed} re-posed, ${f.dropped} dropped (bad camera)`}
        </div>
      )}
      {report.steps != null && (
        <div>
          {report.steps} steps per block at {report.max_resolution} px
        </div>
      )}
      {blocks.length > 0 && (
        <div>
          Held-out PSNR{mean != null && ` (mean ${mean.toFixed(2)} dB)`}:{" "}
          {blocks
            .map(([key, b]) => `${key} ${b.eval ? b.eval.psnr.toFixed(2) : "–"} (${b.train_frames} fr${b.kept ? `, ${(b.kept / 1e6).toFixed(1)}M splats` : ""})`)
            .join(" · ")}
        </div>
      )}
      {sky.length > 0 && (
        <div title="Splats on sky in most views that see them, or more than 10 m above the roofs; held-out PSNR above is from before">
          Sky removed:{" "}
          {sky
            .map(([key, s]) => `${key} ${(((s.sky + s.above_roofs) / s.total) * 100).toFixed(1)}% (${s.sky} on sky, ${s.above_roofs} above roofs)`)
            .join(" · ")}
        </div>
      )}
    </div>
  );
}

function VersionRow({
  v,
  state,
  viewing,
  onView,
  onMakeHead,
  onError,
}: {
  v: Version;
  state: SceneState;
  viewing: boolean;
  onView: () => void;
  onMakeHead: () => void;
  onError: (message: string) => void;
}) {
  const isHead = state.scene.head_version_id === v.id;
  const slug = state.scene.slug;
  const seqs = v.contribution_ids
    .map((id) => state.contributions.find((c) => c.id === id)?.seq)
    .filter((s): s is number => s != null);
  const busy = v.train_status === "queued" || v.train_status === "training" || v.train_status === "cancelling";
  const train = (quality: TrainQuality) =>
    api.train(slug, v.id, quality).catch((err) => onError((err as Error).message));
  const took = TRAIN_OPTIONS.map(({ quality }) => [quality, (v.metrics[`splat_${quality}`] as { seconds?: number } | undefined)?.seconds] as const)
    .filter(([, seconds]) => seconds != null)
    .map(([quality, seconds]) => `${quality} took ${formatDuration(seconds!)}`)
    .join(" · ");
  return (
    <li className={`version ${viewing ? "viewing" : ""}`}>
      <div className="row">
        <span className="title">v{v.number}</span>
        <span className="dots">
          {seqs.map((s) => (
            <span key={s} className="dot" title={`video #${s}`} style={{ background: contributionColor(s) }} />
          ))}
        </span>
        {isHead && <span className="badge done">HEAD</span>}
        {v.dense_url && <span className="badge kind-dense" title="dense point cloud">cloud</span>}
        {v.splat_url && (
          <span className={`badge kind-${v.splat_kind || "none"}`} title={SPLAT_KIND_LABELS[v.splat_kind]}>
            splats{Object.keys(v.splats ?? {}).length ? `: ${Object.keys(v.splats).join(", ")}` : ""}
          </span>
        )}
        <span className="spacer" />
        <button onClick={onView} disabled={viewing}>
          View
        </button>
        {!isHead && (
          <button onClick={onMakeHead} title="Revert/re-apply: make this the scene's current version">
            Make HEAD
          </button>
        )}
      </div>
      {busy && (
        <div className="training">
          <div className="row muted small">
            <span>
              {v.train_status === "queued"
                ? `${v.train_quality} splat training queued${v.train_auto ? " (after any new videos)" : " (starts when no video is processing)"}`
                : v.train_status === "cancelling"
                  ? "cancelling…"
                  : `${v.train_quality} splat training${v.preview_step != null ? ` · live preview @ step ${v.preview_step} (V to watch)` : ""}`}
            </span>
            {v.train_status === "training" && <span className="eta">ETA {formatDuration(v.train_eta)}</span>}
          </div>
          {v.train_status === "training" && (
            <>
              <div className="bar">
                <div style={{ width: `${v.train_progress * 100}%` }} />
              </div>
              {v.train_detail && <div className="muted small">{v.train_detail}</div>}
              {v.train_timeline && v.train_timeline.length > 0 && <Timeline timeline={v.train_timeline} />}
              {v.metrics.training_report != null && (
                <TrainingReport report={v.metrics.training_report as TrainReport}
                  skyRemoved={v.metrics.sky_removed as Record<string, SkyReport> | undefined} />
              )}
            </>
          )}
          <button className="small" onClick={() => api.cancelTraining(slug, v.id).catch((e) => onError(e.message))}>
            Stop training
          </button>
        </div>
      )}
      {!busy && (
        <div className="row small">
          {TRAIN_OPTIONS.filter((o) => !v.splats?.[o.quality]).map((o) => (
            <button key={o.quality} onClick={() => train(o.quality)} title={o.title}>
              {o.label}
            </button>
          ))}
          <span className="muted">
            {took}
            {v.train_status === "failed" && <span className="error"> training failed: {v.train_error?.split("\n")[0]}</span>}
            {!v.train_status && v.train_detail && ` ${v.train_detail}`}
          </span>
        </div>
      )}
      {!busy && (v.metrics.splat_full as TrainReport | undefined)?.blocks && (
        <TrainingReport report={v.metrics.splat_full as TrainReport}
          skyRemoved={v.metrics.sky_removed as Record<string, SkyReport> | undefined} />
      )}
    </li>
  );
}

export function ScenePanel({ state, viewingVersionId, onViewVersion, onError }: Props) {
  const slug = state.scene.slug;
  const contributions = [...state.contributions].reverse();
  const versions = [...state.versions].reverse();
  const viewing = viewingVersionId ?? state.scene.head_version_id;

  return (
    <aside className="panel">
      <h2>{state.scene.name}</h2>
      <Upload slug={slug} onError={onError} />

      <h3>Videos</h3>
      {contributions.length === 0 && <div className="muted">No videos yet.</div>}
      <ul className="contributions">
        {contributions.map((c) => (
          <ContributionCard key={c.id} c={c} />
        ))}
      </ul>

      <h3>Versions</h3>
      {versions.length === 0 && <div className="muted">Nothing published yet.</div>}
      <ul className="versions">
        {versions.map((v) => (
          <VersionRow
            key={v.id}
            v={v}
            state={state}
            viewing={v.id === viewing}
            onView={() => onViewVersion(v.id === state.scene.head_version_id ? null : v.id)}
            onMakeHead={() =>
              api
                .setHead(slug, v.id)
                .then(() => onViewVersion(null))
                .catch((err) => onError((err as Error).message))
            }
            onError={onError}
          />
        ))}
      </ul>
    </aside>
  );
}
