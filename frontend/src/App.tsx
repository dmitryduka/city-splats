import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, contributionColor, SHARED_COLOR, SPLAT_KIND_LABELS, SPLAT_KINDS, useSceneState, type Contribution, type Scene, type SplatKind } from "./api";
import { ReplayBar } from "./components/ReplayBar";
import { SceneList } from "./components/SceneList";
import { ScenePanel } from "./components/ScenePanel";
import { CameraReplay, videoPrefix, type CameraPaths } from "./viewer/cameraPath";
import { SplatViewer, type ViewMode } from "./viewer/SplatViewer";
import type { Geo, MapStyle } from "./viewer/mapTiles";
import { BLOCK_COLORS, BLOCK_LABELS, blockOverlays, type BlockOverlay } from "./viewer/blockOverlay";

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
// Fly speed (scene units/s), remembered per browser.
const SPEED_RANGE = [0.25, 64] as const;
const SPEED_KEY = "flySpeed";
const savedSpeed = () => {
  try {
    const v = Number(localStorage.getItem(SPEED_KEY));
    return v >= SPEED_RANGE[0] && v <= SPEED_RANGE[1] ? v : 4;
  } catch {
    return 4;
  }
};

type Toast = { id: number; text: string; kind: "info" | "error" };

/** A boolean setting remembered per browser (as long as storage works). */
function useRemembered(key: string, initial: boolean) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored === null ? initial : stored === "1";
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, value ? "1" : "0");
    } catch {
      // private window: not remembered
    }
  }, [key, value]);
  return [value, setValue] as const;
}

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
  const [showBlocks, setShowBlocks] = useState(true);
  // Place names (L) and street names (N) over GPS-placed scenes.
  const [showPlaces, setShowPlaces] = useRemembered("showPlaces", true);
  const [showStreets, setShowStreets] = useRemembered("showStreets", false);
  // Collapsed to a thin strip: the scene list, the HUD, the scene panel.
  const [listCollapsed, setListCollapsed] = useRemembered("collapsed:list", false);
  const [hudCollapsed, setHudCollapsed] = useRemembered("collapsed:hud", false);
  const [panelCollapsed, setPanelCollapsed] = useRemembered("collapsed:panel", false);
  const [mapStyle, setMapStyle] = useState<MapStyle | null>(null);
  // Cloud/splat opacity, offered while a map is shown so it can show through. Applied cubed:
  // many splats overlap along each ray, so per-splat opacity must drop fast to see through.
  const [opacity, setOpacity] = useState(1);
  // The map's height by eye, metres, remembered per scene.
  const [mapLift, setMapLift] = useState(0);
  useEffect(() => {
    try {
      setMapLift(Number(localStorage.getItem(`mapLift:${slug}`) ?? 0));
    } catch {
      setMapLift(0);
    }
  }, [slug]);
  const changeMapLift = (metres: number) => {
    setMapLift(metres);
    try {
      localStorage.setItem(`mapLift:${slug}`, String(metres));
    } catch {
      // storage unavailable: not remembered
    }
  };
  const [speed, setSpeed] = useState(savedSpeed);
  useEffect(() => {
    try {
      localStorage.setItem(SPEED_KEY, String(speed));
    } catch {
      // storage unavailable: the speed just isn't remembered
    }
  }, [speed]);
  const [mapSource, setMapSource] = useState<{ attribution: string; failed: boolean } | null>(null);
  const geoRef = useRef<Geo | null>(null);
  const [fps, setFps] = useState(0);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const onStats = useCallback((s: { fps: number }) => setFps(s.fps), []);
  // Replaying a video's camera path (Play on its card); the speed is kept between replays.
  const [replay, setReplay] = useState<{ replay: CameraReplay; label: string } | null>(null);
  const replayRef = useRef(replay);
  replayRef.current = replay;
  const [replaySpeed, setReplaySpeed] = useState(1);
  const [cameraPaths, setCameraPaths] = useState<CameraPaths | null>(null);

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
  useEffect(() => {
    setViewingVersionId(null);
    setReplay(null);
  }, [slug]);

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
      if (e.key === "Escape") setReplay(null);
      if ((e.target as HTMLElement).tagName === "INPUT") return;
      if ((e.key === "p" || e.key === "P") && replayRef.current) replayRef.current.replay.paused = !replayRef.current.replay.paused;
      if (MODE_KEYS[e.key]) show(MODE_KEYS[e.key]);
      if (e.key === "v" || e.key === "V") show(modeRef.current === "splats" ? "dense" : "splats");
      if (e.key === "c" || e.key === "C") setShowCameras((v) => !v);
      if (e.key === "b" || e.key === "B") {
        if (!geoRef.current) toast("No blocks to show: none of this version's videos has GPS", "error");
        else setShowBlocks((v) => !v);
      }
      if (e.key === "l" || e.key === "L") {
        if (!geoRef.current) toast("No place names: none of this version's videos has GPS", "error");
        else setShowPlaces((v) => !v);
      }
      if (e.key === "n" || e.key === "N") {
        if (!geoRef.current) toast("No street names: none of this version's videos has GPS", "error");
        else setShowStreets((v) => !v);
      }
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
  // Written next to the layers (worker.pipeline.camera_paths); older versions may lack it.
  useEffect(() => {
    setCameraPaths(null);
    if (!layersUrl) return;
    let cancelled = false;
    fetch(`${layersUrl}/camera_paths.json`)
      .then((r) => (r.ok ? (r.json() as Promise<CameraPaths>) : null))
      .then((paths) => !cancelled && setCameraPaths(paths))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [layersUrl]);
  const pathOf = (c: Contribution) => {
    const prefix = videoPrefix(c.video_path);
    const rows = prefix === "fov" ? null : cameraPaths?.[prefix];
    return rows && rows.length >= 2 ? rows : null;
  };
  const play = (c: Contribution) => {
    const rows = pathOf(c);
    if (!rows) return;
    const next = new CameraReplay(rows, cameraPaths?.fov?.[videoPrefix(c.video_path)] ?? null);
    next.speed = replaySpeed;
    setReplay({ replay: next, label: `#${c.seq} ${c.filename}` });
  };
  // Parsed from its JSON so state updates (a new object each time) don't rebuild the map.
  const geoJson = JSON.stringify(selected?.metrics.geo ?? null);
  const geo = useMemo(() => JSON.parse(geoJson) as Geo | null, [geoJson]);
  geoRef.current = geo;
  useEffect(() => setMapSource(null), [mapStyle, geo]);
  const training = selected?.train_status === "training" || selected?.train_status === "cancelling";
  const blocksJson = JSON.stringify(selected && geo ? blockOverlays(selected.metrics, training, kind) : []);
  const blocks = useMemo(() => JSON.parse(blocksJson) as BlockOverlay[], [blocksJson]);
  const sizable = surface === "dense" || (!live && kind === "uniform");
  const seqs = new Set(
    (selected?.contribution_ids ?? [])
      .map((id) => state?.contributions.find((c) => c.id === id)?.seq)
      .filter((s): s is number => s != null),
  );

  return (
    <div className="layout" style={{ gridTemplateColumns: `${listCollapsed ? "24px" : "220px"} 1fr ${panelCollapsed ? "24px" : "340px"}` }}>
      {/* Collapsed sides stay mounted (hidden): the panel keeps uploads and their progress. */}
      <div className={`side left${listCollapsed ? " collapsed" : ""}`}>
        <button className="collapse" onClick={() => setListCollapsed((v) => !v)} title={listCollapsed ? "Show scenes" : "Hide scenes"}>
          {listCollapsed ? "›" : "‹"}
        </button>
        <SceneList
          scenes={scenes}
          current={slug}
          onSelect={(s) => (window.location.hash = `#/scenes/${s}`)}
          onCreated={(scene) => {
            refreshScenes();
            window.location.hash = `#/scenes/${scene.slug}`;
          }}
        />
      </div>
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
          splatOpacity={mapStyle && geo ? opacity ** 3 : 1}
          moveSpeed={speed}
          onStats={onStats}
          geo={geo}
          mapStyle={mapStyle}
          mapLift={mapLift}
          showPlaces={showPlaces}
          showStreets={showStreets}
          onMapSource={(attribution, failed) => setMapSource({ attribution, failed })}
          blocks={showBlocks ? blocks : null}
          replay={replay?.replay ?? null}
          onReplayEnd={() => setReplay(null)}
        />}
        {replay && (
          <ReplayBar
            replay={replay.replay}
            label={replay.label}
            onSpeed={(s) => {
              replay.replay.speed = s;
              setReplaySpeed(s);
            }}
            onStop={() => setReplay(null)}
          />
        )}
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
        <div className={`hud${hudCollapsed ? " collapsed" : ""}`} data-labels-avoid>
          <div className="hud-title">
            <button className="collapse" onClick={() => setHudCollapsed((v) => !v)} title={hudCollapsed ? "Show controls" : "Hide controls"}>
              {hudCollapsed ? "▸" : "▾"}
            </button>
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
          {hudCollapsed ? (
            <div className="hud-line">{fps < 0.5 ? "idle" : `${fps.toFixed(0)} fps`} · {MODE_LABELS[mode]}</div>
          ) : (<>
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
          <label className="size-slider" title="WASD / QE speed (hold Shift for 5x)">
            fly speed
            <input
              type="range"
              min={Math.log(SPEED_RANGE[0])}
              max={Math.log(SPEED_RANGE[1])}
              step={0.01}
              value={Math.log(speed)}
              onChange={(e) => setSpeed(Math.exp(Number(e.target.value)))}
            />
            <span>{speed < 10 ? speed.toFixed(1) : speed.toFixed(0)}/s</span>
          </label>
          {mapStyle && geo && (mode === "dense" || mode === "splats") && (
            <label className="size-slider" title="See the map through the scene">
              opacity
              <input type="range" min={0} max={1} step={0.01} value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))} />
              <span>{Math.round(opacity * 100)}%</span>
            </label>
          )}
          {mapStyle && geo && (
            <label className="size-slider" title="Raise or lower the map to line it up with the scene's ground">
              map height
              <input type="range" min={-40} max={20} step={0.5} value={mapLift}
                onChange={(e) => changeMapLift(Number(e.target.value))} />
              <span>{mapLift > 0 ? "+" : ""}{mapLift.toFixed(1)} m</span>
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
          {showBlocks && blocks.length > 0 && (
            <div className="legend">
              training blocks:
              {(["done", "reused", "training", "queued", "stale"] as const)
                .filter((s) => s === "done" || blocks.some((b) => b.status === s))
                .map((s) => (
                  <span key={s}>
                    <i style={{ background: BLOCK_COLORS[s], borderRadius: 1 }} />
                    {BLOCK_LABELS[s]} {blocks.filter((b) => b.status === s).length}
                  </span>
                ))}
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
          <div className="muted">WASD / drag · V cloud ⇄ splats · 1 cloud · 2 splats · 3 by video · 4 sparse · K splat kind · C cameras · M map · L place names · N street names · B blocks · [ ] splat size · ▶ on a video replays its path (P pause · Esc stop)</div>
          <label className="follow">
            <input type="checkbox" checked={followLive} onChange={(e) => setFollowLive(e.target.checked)} /> follow
            live training
          </label>
          </>)}
        </div>
        <div className="toasts">
          {toasts.map((t) => (
            <div key={t.id} className={`toast ${t.kind}`}>
              {t.text}
            </div>
          ))}
        </div>
      </main>
      <div className={`side right${panelCollapsed ? " collapsed" : ""}`}>
        <button className="collapse" onClick={() => setPanelCollapsed((v) => !v)} title={panelCollapsed ? "Show videos and versions" : "Hide videos and versions"}>
          {panelCollapsed ? "‹" : "›"}
        </button>
        {state && (
          <ScenePanel
            state={state}
            viewingVersionId={viewingVersionId}
            onViewVersion={setViewingVersionId}
            canReplay={(c) => pathOf(c) != null}
            onReplay={play}
            onError={(m) => toast(m, "error")}
          />
        )}
      </div>
    </div>
  );
}
