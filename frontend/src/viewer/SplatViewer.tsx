import { useEffect, useRef } from "react";
import * as THREE from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { dyno, SparkControls, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { buildMap, type Geo, type MapStyle } from "./mapTiles";

/** "dense" and "splats" both show the streamed meshes at splatUrl (a dense cloud is streamed
 * as small splats); the points modes show the sparse PLY layers instead. */
export type ViewMode = "dense" | "splats" | "points_source" | "points_rgb";

type Props = {
  /** URL of a (chunked) .rad LoD file, or of a tile list (.json: {tiles: [{url}]}, one .rad per
   * tile); streamed via Spark's pager. Swapped in place on change. */
  splatUrl: string | null;
  /** Directory URL with points_source.ply, points_rgb.ply and cameras.ply. */
  layersUrl: string | null;
  mode: ViewMode;
  showCameras: boolean;
  /** Render pixel ratio cap (Retina = 2 is ~4x the fragment work of 1). */
  pixelRatio?: number;
  /** Spark LoD splat budget multiplier (1 = platform default). */
  lodSplatScale?: number;
  /** Multiplier on every splat's size (1 = as built/trained); for clouds and uniform splats. */
  splatSize?: number;
  /** Opacity of the cloud/splats (1 = as built), e.g. to see the map underneath. */
  splatOpacity?: number;
  /** Fly speed for WASD/QE, scene units per second (Shift: 5x). */
  moveSpeed?: number;
  onStats?: (stats: { fps: number }) => void;
  /** Where the scene lies on the map (from GPS), and which map to lay under it (null: none). */
  geo?: Geo | null;
  mapStyle?: MapStyle | null;
  /** The map's attribution once tiles load; failed when no provider served a tile. */
  onMapSource?: (attribution: string, failed: boolean) => void;
};

type PointLayer = "points_source" | "points_rgb" | "cameras";

type Context = {
  world: THREE.Group;
  /** One mesh, or one per tile; Spark picks LoD across all of them under one budget. */
  splats: SplatMesh[];
  /** Shader uniform behind splatSize, shared by every mesh this viewer creates. */
  splatSize: dyno.DynoFloat<string>;
  points: Partial<Record<PointLayer, THREE.Points>>;
  /** Something visible changed; render on the next animation frame. */
  dirty: boolean;
};

/** Scales every splat by the size uniform at render time, so changing it needs no reload. */
function sizeModifier(size: dyno.DynoFloat<string>) {
  return dyno.dynoBlock({ gsplat: dyno.Gsplat }, { gsplat: dyno.Gsplat }, ({ gsplat }) => {
    const { scales } = dyno.splitGsplat(gsplat!).outputs;
    return { gsplat: dyno.combineGsplat({ gsplat, scales: dyno.mul(scales, size) }) };
  });
}

function disposeMeshes(meshes: SplatMesh[]) {
  for (const mesh of meshes) {
    mesh.removeFromParent();
    mesh.dispose();
  }
}

/** The .rad URLs behind splatUrl: itself, or a tile list's entries. */
async function lodUrls(splatUrl: string): Promise<string[]> {
  if (!new URL(splatUrl, location.href).pathname.endsWith(".json")) return [splatUrl];
  const response = await fetch(splatUrl);
  if (!response.ok) throw new Error(`${splatUrl}: ${response.status}`);
  const { tiles } = (await response.json()) as { tiles: { url: string }[] };
  return tiles.map((t) => t.url);
}

function disposePoints(points: THREE.Points | undefined) {
  points?.removeFromParent();
  points?.geometry.dispose();
  (points?.material as THREE.Material | undefined)?.dispose();
}

/**
 * The renderer, camera and controls live as long as the component; splat and layer URLs are
 * swapped underneath without resetting the camera, so live updates don't jolt the view.
 *
 * Renders on demand: only when the camera moved, Spark reports new work (sort/LoD/streamed
 * chunk via onDirty) or the content changed. An idle tab then leaves the GPU to the splat
 * trainer instead of redrawing the same frame at 60 fps.
 */
export function SplatViewer({
  splatUrl,
  layersUrl,
  mode,
  showCameras,
  pixelRatio = 1,
  lodSplatScale = 1,
  splatSize = 1,
  splatOpacity = 1,
  moveSpeed = 1,
  onStats,
  geo = null,
  mapStyle = null,
  onMapSource,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<Context | null>(null);
  const viewRef = useRef({ mode, showCameras });
  const onStatsRef = useRef(onStats);
  onStatsRef.current = onStats;
  const controlsRef = useRef<SparkControls | null>(null);
  const moveSpeedRef = useRef(moveSpeed);
  moveSpeedRef.current = moveSpeed;
  const opacityRef = useRef(splatOpacity);
  opacityRef.current = splatOpacity;
  const onMapSourceRef = useRef(onMapSource);
  onMapSourceRef.current = onMapSource;

  const applyVisibility = () => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const { mode, showCameras } = viewRef.current;
    for (const mesh of ctx.splats) mesh.visible = mode === "dense" || mode === "splats";
    if (ctx.points.points_source) ctx.points.points_source.visible = mode === "points_source";
    if (ctx.points.points_rgb) ctx.points.points_rgb.visible = mode === "points_rgb";
    if (ctx.points.cameras) ctx.points.cameras.visible = showCameras;
    ctx.dirty = true;
  };

  // Renderer lifetime.
  useEffect(() => {
    const container = containerRef.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, pixelRatio));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111318);
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.01, 1000);

    const ctx: Context = {
      world: new THREE.Group(),
      splats: [],
      splatSize: dyno.dynoFloat(splatSize),
      points: {},
      dirty: true,
    };
    scene.add(new SparkRenderer({ renderer, lodSplatScale, onDirty: () => (ctx.dirty = true) }));

    // Reconstructions live in the COLMAP/OpenCV frame (y down, z forward); flip into
    // three.js (y up, z backward) once at the root.
    ctx.world.quaternion.set(1, 0, 0, 0);
    scene.add(ctx.world);
    ctxRef.current = ctx;

    const controls = new SparkControls({ canvas: renderer.domElement });
    controls.fpsMovement.moveSpeed = moveSpeedRef.current;
    controlsRef.current = controls;
    const observer = new ResizeObserver(() => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
      ctx.dirty = true;
    });
    observer.observe(container);

    let frames = 0;
    let last = performance.now();
    const lastPose = new THREE.Matrix4();
    renderer.setAnimationLoop(() => {
      controls.update(camera);
      camera.updateMatrixWorld();
      const moved = !camera.matrixWorld.equals(lastPose);
      if (moved || ctx.dirty) {
        ctx.dirty = false;
        lastPose.copy(camera.matrixWorld);
        renderer.render(scene, camera);
        frames++;
      }
      const now = performance.now();
      if (now - last > 1000) {
        onStatsRef.current?.({ fps: (frames * 1000) / (now - last) });
        frames = 0;
        last = now;
      }
    });

    return () => {
      renderer.setAnimationLoop(null);
      observer.disconnect();
      const ctx = ctxRef.current;
      ctx?.splats.forEach((mesh) => mesh.dispose());
      Object.values(ctx?.points ?? {}).forEach(disposePoints);
      ctxRef.current = null;
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, [pixelRatio, lodSplatScale]);

  // Splat swap: add the new meshes, drop the old ones once all new ones are initialized.
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    if (!splatUrl) {
      // Nothing to show (e.g. a version without trained splats): don't keep the last one.
      disposeMeshes(ctx.splats);
      ctx.splats = [];
      ctx.dirty = true;
      return;
    }
    let cancelled = false;
    let next: SplatMesh[] = [];
    lodUrls(splatUrl)
      .then((urls) => {
        if (cancelled) return;
        next = urls.map((url) => {
          const mesh = new SplatMesh({ url, paged: true, objectModifier: sizeModifier(ctx.splatSize) });
          mesh.visible = false;
          mesh.opacity = opacityRef.current;
          ctx.world.add(mesh);
          return mesh;
        });
        return Promise.all(next.map((mesh) => mesh.initialized));
      })
      .then(() => {
        if (cancelled) return;
        disposeMeshes(ctx.splats);
        ctx.splats = next;
        applyVisibility();
      })
      .catch((error) => console.error("loading splats failed", error));
    return () => {
      cancelled = true;
      if (ctx.splats !== next) disposeMeshes(next);
    };
  }, [splatUrl, pixelRatio, lodSplatScale]);

  // Point layers swap.
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !layersUrl) return;
    let cancelled = false;
    const sizes: Record<PointLayer, number> = { points_source: 2, points_rgb: 2, cameras: 7 };
    for (const name of Object.keys(sizes) as PointLayer[]) {
      new PLYLoader().loadAsync(`${layersUrl}/${name}.ply`).then((geometry) => {
        if (cancelled) return geometry.dispose();
        const material = new THREE.PointsMaterial({ size: sizes[name], vertexColors: true, sizeAttenuation: false });
        const points = new THREE.Points(geometry, material);
        disposePoints(ctx.points[name]);
        ctx.points[name] = points;
        ctx.world.add(points);
        applyVisibility();
      });
    }
    return () => {
      cancelled = true;
    };
  }, [layersUrl, pixelRatio, lodSplatScale]);

  // Map tiles under the scene.
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || !geo || !mapStyle) return;
    const map = buildMap(geo, mapStyle, () => (ctx.dirty = true), (a, failed) => onMapSourceRef.current?.(a, failed));
    ctx.world.add(map.group);
    ctx.dirty = true;
    return () => {
      map.dispose();
      ctx.dirty = true;
    };
  }, [geo, mapStyle, pixelRatio, lodSplatScale]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx || ctx.splatSize.value === splatSize) return;
    ctx.splatSize.value = splatSize;
    ctx.splats.forEach((mesh) => mesh.updateVersion());
    ctx.dirty = true;
  }, [splatSize]);

  useEffect(() => {
    if (controlsRef.current) controlsRef.current.fpsMovement.moveSpeed = moveSpeed;
  }, [moveSpeed]);

  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    for (const mesh of ctx.splats) {
      mesh.opacity = splatOpacity;
      mesh.updateVersion();
    }
    ctx.dirty = true;
  }, [splatOpacity]);

  useEffect(() => {
    viewRef.current = { mode, showCameras };
    applyVisibility();
  }, [mode, showCameras]);

  return <div ref={containerRef} style={{ position: "absolute", inset: 0 }} />;
}
