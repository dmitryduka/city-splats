import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, contributionColor, SHARED_COLOR, SPLAT_KIND_LABELS, SPLAT_KINDS, useSceneState, type Scene, type SplatKind } from "./api";
import { SceneList } from "./components/SceneList";
import { ScenePanel } from "./components/ScenePanel";
import { SplatViewer, type ViewMode } from "./viewer/SplatViewer";
import type { Geo, MapStyle } from "./viewer/mapTiles";

const MODE_KEYS: Record<string, ViewMode> = { "1": "dense", "2": "splats", "3": "points_source", "4": "points_rgb" };
const MODE_LABELS: Record<ViewMode, string> = {
  dense: "dense point cloud",
  splats: "splats",
  points_source: "points by video",
  points_rgb: "points (color)",
};

// Splat size multiplier steps ([ and ]), for clouds and uniform splats.
const SIZE_STEP = 1.15;
// M cycles the map under a GPS-placed scene: off -> street map -> satellite -> off.
const MAP_CYCLE: (MapStyle | null)[] = [null, "street", "satellite"];
const clampSize = (s: number) => Math.min(4, Math.max(0.25, s));

type Toast = { id: number; text: string; kind: "info" | "error" };

const slugFromHash = () => window.location.hash.match(/^#\/scenes\/([^/]+)/)?.[1] ?? null;

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const pixelRatio = Number(params.get("pr") ?? 1);
  const lodSplatScale = Number(params.get("lod") ?? 1);

  const [scenes, setScenes] = useState<Scene[]>([]);
  const [slug, setSlug] = useState<string | null>(slugFromHash());
  const { state, connected } = useSceneState(slug);
  const [viewingVersionId, setViewingVersionId] = useState<number | null>(null);
  const [followLive, setFollowLive] = useState(true);
  const [mode, setMode] = useState<ViewMode>("dense");
  // The surface (dense cloud or splats) stays loaded while a sparse-points mode is shown.
  const [surface, setSurface] = useState<"dense" | "splats">("dense");
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const show = useCallback((next: ViewMode) => {
    setMode(next);
    if (next === "dense" || next === "splats") setSurface(next);
  }, []);
  // Render-time size multiplier per surface; only offered where splats share one size.
  const [sizes, setSizes] = useState({ dense: 1, splats: 1 });
  const surfaceRef = useRef(surface);
  surfaceRef.current = surface;
  const resize = useCallback((factor: number) => {
    setSizes((s) => ({ ...s, [surfaceRef.current]: clampSize(s[surfaceRef.current] * factor) }));
  }, []);
  // Which trained splats to show (null: the latest trained).
  const [kindChoice, setKindChoice] = useState<SplatKind | null>(null);
  const kindsRef = useRef<{ kinds: SplatKind[]; kind: SplatKind | null }>({ kinds: [], kind: null });
  const [showCameras, setShowCameras] = useState(false);
  const [mapStyle, setMapStyle] = useState<MapStyle | null>(null);
  const [mapSource, setMapSource] = useState<{ attribution: string; failed: boolean } | null>(null);
  const geoRef = useRef<Geo | null>(null);
  const [fps, setFps] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const onStats = useCallback((s: { fps: number }) => setFps(s.fps), []);

  const toast = useCallback((text: string, kind: Toast["kind"] = "info") => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 8000);
  }, []);

  const refreshScenes = useCallback(() => api.listScenes().then(setScenes).catch(() => {}), []);
  useEffect(() => {
    refreshScenes();
    const timer = setInterval(refreshScenes, 10000);
    return () => clearInterval(timer);
  }, [refreshScenes]);

  useEffect(() => {
    const onHash = () => setSlug(slugFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    if (!slug && scenes.length) window.location.hash = `#/scenes/${scenes[0].slug}`;
  }, [slug, scenes]);
  useEffect(() => setViewingVersionId(null), [slug]);

  // Notifications: new HEAD, rejected/failed videos.
  const seen = useRef<{ slug: string | null; head: number | null; kind: string; statuses: Map<number, string> }>({
    slug: null,
    head: null,
    kind: "",
    statuses: new Map(),
  });
  useEffect(() => {
    if (!state) return;
    const prev = seen.current;
    const sameScene = prev.slug === state.scene.slug;
    if (sameScene && state.head && prev.head !== state.head.id) {
      const c = state.contributions.find((x) => x.id === state.head!.contribution_id);
      toast(`Scene updated → v${state.head.number}${c ? ` (video #${c.seq} ${c.filename})` : ""}`);
      refreshScenes();
    } else if (sameScene && state.head && prev.kind !== state.head.splat_kind) {
      if (state.head.splat_kind && state.head.splat_kind !== "points" && state.head.splat_kind !== "draft")
      {
        setKindChoice(null);
        toast(`v${state.head.number}: ${SPLAT_KIND_LABELS[state.head.splat_kind]} ready, press V to switch`);
      }
    }
    for (const c of state.contributions) {
      const before = prev.statuses.get(c.id);
      if (sameScene && before && before !== c.status && (c.status === "rejected" || c.status === "failed")) {
        toast(`Video #${c.seq} ${c.status}: ${c.error?.split("\n")[0] ?? ""}`, "error");
      }
    }
    seen.current = {
      slug: state.scene.slug,
      head: state.head?.id ?? null,
      kind: state.head?.splat_kind ?? "",
      statuses: new Map(state.contributions.map((c) => [c.id, c.status])),
    };
  }, [state, toast, refreshScenes]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if (MODE_KEYS[e.key]) show(MODE_KEYS[e.key]);
      if (e.key === "v" || e.key === "V") show(modeRef.current === "splats" ? "dense" : "splats");
      if (e.key === "c" || e.key === "C") setShowCameras((v) => !v);
      if (e.key === "m" || e.key === "M") {
        if (!geoRef.current) toast("No map: none of this version's videos has GPS", "error");
        else setMapStyle((s) => MAP_CYCLE[(MAP_CYCLE.indexOf(s) + 1) % MAP_CYCLE.length]);
      }
      if (e.key === "k" || e.key === "K") {
        const { kinds, kind } = kindsRef.current;
        if (kinds.length > 1) {
          setKindChoice(kinds[(kinds.indexOf(kind!) + 1) % kinds.length]);
          show("splats");
        }
      }
      if (e.key === "[") resize(1 / SIZE_STEP);
      if (e.key === "]") resize(SIZE_STEP);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [show, resize, toast]);

  // What to show: the dense cloud, or the splats (the live training preview while training,
  // if following, else the trained splat).
  const selected = state?.versions.find((v) => v.id === viewingVersionId) ?? state?.head ?? null;
  const live = surface === "splats" && followLive && selected?.preview_url ? selected : null;
  const hasSplats = Boolean(selected?.splat_url || selected?.preview_url);
  const kinds = SPLAT_KINDS.filter((k) => selected?.splats?.[k]);
  const kind = kindChoice && kinds.includes(kindChoice) ? kindChoice : selected?.splat_kind || null;
  kindsRef.current = { kinds, kind };
  const splatUrl =
    surface === "dense"
      ? selected?.dense_url || null
      : live?.preview_url || (kind && selected?.splats?.[kind]) || selected?.splat_url || null;
  const layersUrl = selected?.layers_url ?? null;
  // Parsed from its JSON so state updates (a new object each time) don't rebuild the map.
  const geoJson = JSON.stringify(selected?.metrics.geo ?? null);
  const geo = useMemo(() => JSON.parse(geoJson) as Geo | null, [geoJson]);
  geoRef.current = geo;
  useEffect(() => setMapSource(null), [mapStyle, geo]);
  const sizable = surface === "dense" || (!live && kind === "uniform");
  const seqs = new Set(
    (selected?.contribution_ids ?? [])
      .map((id) => state?.contributions.find((c) => c.id === id)?.seq)
      .filter((s): s is number => s != null),
  );

  return (
    <div className="layout">
      <SceneList
        scenes={scenes}
        current={slug}
        onSelect={(s) => (window.location.hash = `#/scenes/${s}`)}
        onCreated={(scene) => {
          refreshScenes();
          window.location.hash = `#/scenes/${scene.slug}`;
        }}
      />
      <main className="viewport">
        {slug && <SplatViewer
          key={slug}
          splatUrl={splatUrl}
          layersUrl={layersUrl}
          mode={mode}
          showCameras={showCameras}
          pixelRatio={pixelRatio}
          lodSplatScale={lodSplatScale}
          splatSize={sizable ? sizes[surface] : 1}
          onStats={onStats}
          geo={geo}
          mapStyle={mapStyle}
          onMapSource={(attribution, failed) => setMapSource({ attribution, failed })}
        />}
        {state && !splatUrl && !selected && (
          <div className="empty">Upload a video to start reconstructing “{state.scene.name}”.</div>
        )}
        {selected && !splatUrl && mode === "splats" && (
          <div className="empty">
            {selected.train_status === "queued" || selected.train_status === "training"
              ? "Splat training is starting… the live preview appears after the first 1000 steps."
              : "No splats for this version yet: press “Train splats” in the panel, or V for the point cloud."}
          </div>
        )}
        {selected && !splatUrl && mode === "dense" && (
          <div className="empty">This version has no dense point cloud (made before dense clouds). Press V for splats or 3 / 4 for sparse points.</div>
        )}
        <div className="hud">
          <div>
            {live ? (
              <span className="live">
                ● LIVE {live.train_quality} training · v{live.number} · step {live.preview_step}
              </span>
            ) : selected ? (
              <>
                v{selected.number}
                {selected.id === state?.scene.head_version_id ? " (HEAD)" : " (viewing older version)"}
                {selected.train_status === "training" && ` · training splats ${Math.round(selected.train_progress * 100)}%`}
              </>
            ) : (
              "no version yet"
            )}
          </div>
          <div className="surface-switch">
            <button className={mode === "dense" ? "on" : ""} onClick={() => show("dense")} title="1 or V">
              Point cloud
            </button>
            <button
              className={mode === "splats" ? "on" : ""}
              onClick={() => show("splats")}
              title={hasSplats ? "2 or V" : "No splats yet: train them from the version in the panel"}
            >
              Splats{hasSplats ? "" : " (none)"}
            </button>
          </div>
          {mode === "splats" && !live && kinds.length > 1 && (
            <div className="surface-switch kinds">
              {kinds.map((k) => (
                <button key={k} className={k === kind ? "on" : ""} onClick={() => setKindChoice(k)} title={`${SPLAT_KIND_LABELS[k]} (K cycles)`}>
                  {k}
                </button>
              ))}
            </div>
          )}
          {sizable && (mode === "dense" || mode === "splats") && (
            <label className="size-slider" title="[ smaller · ] bigger">
              splat size
              <input
                type="range"
                min={Math.log(0.25)}
                max={Math.log(4)}
                step={0.01}
                value={Math.log(sizes[surface])}
                onChange={(e) => setSizes((s) => ({ ...s, [surface]: clampSize(Math.exp(Number(e.target.value))) }))}
              />
              <span>{sizes[surface].toFixed(2)}×</span>
            </label>
          )}
          <div>
            {fps < 0.5 ? "idle (renders on demand)" : `${fps.toFixed(0)} fps`} · {MODE_LABELS[mode]}
            {showCameras ? " + cameras" : ""} {connected ? "" : "· reconnecting…"}
          </div>
          {mapStyle && geo && (
            <div className={mapSource?.failed ? "map-note error" : "map-note"}>
              {mapStyle === "street" ? "street map" : "satellite"} · placed from GPS (±{geo.error_m} m)
              {mapSource && (mapSource.failed ? " · some map tiles failed to load" : ` · ${mapSource.attribution}`)}
            </div>
          )}
          {mode === "points_source" && (
            <div className="legend">
              {[...seqs].map((s) => (
                <span key={s}>
                  <i style={{ background: contributionColor(s) }} />#{s}
                </span>
              ))}
              <span>
                <i style={{ background: SHARED_COLOR }} />
                shared
              </span>
            </div>
          )}
          <div className="muted">WASD / drag · V cloud ⇄ splats · 1 cloud · 2 splats · 3 by video · 4 sparse · K splat kind · C cameras · M map · [ ] splat size</div>
          <label className="follow">
            <input type="checkbox" checked={followLive} onChange={(e) => setFollowLive(e.target.checked)} /> follow
            live training
          </label>
        </div>
        <div className="toasts">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.kind}`}>
              {t.text}
            </div>
          ))}
        </div>
      </main>
      {state && (
        <ScenePanel
          state={state}
          viewingVersionId={viewingVersionId}
          onViewVersion={setViewingVersionId}
          onError={(m) => toast(m, "error")}
        />
      )}
    </div>
  );
}
