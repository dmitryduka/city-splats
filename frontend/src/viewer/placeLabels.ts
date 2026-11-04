import * as THREE from "three";
import { loadHeights, type Geo, type Heights } from "./mapTiles";
import type { Place } from "./places";

const EARTH_RADIUS = 6_378_137; // as worker.pipeline.devices.dji
/** Places farther than this from the scene's origin belong to another city. */
const RANGE_M = 30_000;
/** Closer than NEAR_M a label shows its note; beyond FAR_M only landmarks (major) are shown,
 * up to MAJOR_FAR_M. */
const NEAR_M = 800;
const FAR_M = 3_000;
const MAJOR_FAR_M = 15_000;
const MAX_SHOWN = 30;
/** A label appears once it has had room for this long, so flying doesn't make them flicker. */
const SETTLE_MS = 250;
/** Pixels: the stem's height above the dot, and the free border around the screen. */
const STEM = 20;
const MARGIN = 4;

type Side = "r" | "l";
type Rect = { x0: number; x1: number; y0: number; y1: number };

type Label = {
  place: Place;
  /** The dot, in the map frame (metres east/north, z above the fitted ground plane). */
  p: THREE.Vector3;
  el: HTMLDivElement;
  txt: HTMLDivElement;
  on: boolean;
  /** When it first had room (performance.now()), -1 while it hasn't. */
  since: number;
  side: Side;
  near: boolean;
  /** Text size [w, h] as measured, without and with the note. */
  dims: Partial<Record<"far" | "near", [number, number]>>;
};

export type PlaceLabels = {
  /** Place the labels for this camera view; true while one is waiting to appear (call again). */
  update: (camera: THREE.PerspectiveCamera) => boolean;
  setLift: (metres: number) => void;
  dispose: () => void;
};

const overlaps = (a: Rect, b: Rect) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/**
 * Names over places on the map, drawn as HTML over the canvas: a dot at the place, a stem
 * and the name above it (after p.migdal.pl/warsaw-vibe-voxel). Landmarks first, then by
 * distance, each label takes the first free side (right, then left) of its dot; one that
 * would overlap another, or a panel over the view (marked data-labels-avoid), stays hidden. The splats aren't hit-tested, so a label shows through
 * whatever stands in front of its place.
 */
export function buildPlaceLabels(
  geo: Geo,
  places: Place[],
  world: THREE.Object3D,
  container: HTMLElement,
  onChange: () => void,
): PlaceLabels {
  const root = document.createElement("div");
  root.className = "place-labels";
  container.appendChild(root);

  const [lat0, lon0] = geo.origin;
  const rad = Math.PI / 180;
  const labels: Label[] = [];
  for (const place of places) {
    const east = (place.lon - lon0) * rad * EARTH_RADIUS * Math.cos(lat0 * rad);
    const north = (place.lat - lat0) * rad * EARTH_RADIUS;
    if (Math.hypot(east, north) > RANGE_M) continue;
    const el = document.createElement("div");
    el.className = `place off${place.major ? " major" : ""}`;
    el.innerHTML = `<i class="dot"></i><i class="stem"></i><div class="txt"><b></b><span></span></div>`;
    const txt = el.querySelector<HTMLDivElement>(".txt")!;
    txt.querySelector("b")!.textContent = place.name;
    txt.querySelector("span")!.textContent = place.note ?? "";
    root.appendChild(el);
    labels.push({ place, p: new THREE.Vector3(east, north, place.h ?? 0), el, txt, on: false, since: -1, side: "r", near: false, dims: {} });
  }

  // On the terrain once it loads (the map's own grid); until then hidden, not misplaced.
  let ready = !geo.terrain;
  let disposed = false;
  const setHeights = (heights: Heights) => {
    for (const l of labels) l.p.z = heights(l.p.x, l.p.y) + (l.place.h ?? 0);
  };
  if (geo.terrain)
    loadHeights(geo.terrain).then((heights) => {
      if (disposed) return;
      setHeights(heights);
      ready = true;
      onChange();
    }).catch((e) => console.warn("terrain not loaded, labels stay hidden:", e));

  const sceneFromMap = new THREE.Matrix4().set(...(geo.scene_from_map.flat() as Parameters<THREE.Matrix4["set"]>));
  const worldFromMap = new THREE.Matrix4();
  const mapFromWorld = new THREE.Matrix4();
  let lift = 0;
  const v = new THREE.Vector3();
  const eye = new THREE.Vector3();

  const show = (l: Label, on: boolean) => {
    if (l.on !== on) l.el.classList.toggle("off", !on);
    l.on = on;
  };

  const update = (camera: THREE.PerspectiveCamera): boolean => {
    if (!ready) return false;
    const width = container.clientWidth, height = container.clientHeight;
    world.updateWorldMatrix(true, false);
    worldFromMap.multiplyMatrices(world.matrixWorld, sceneFromMap).multiply(new THREE.Matrix4().makeTranslation(0, 0, lift));
    mapFromWorld.copy(worldFromMap).invert();
    eye.copy(camera.position).applyMatrix4(mapFromWorld);
    const now = performance.now();
    const box = container.getBoundingClientRect();
    const taken: Rect[] = [...(container.parentElement?.querySelectorAll("[data-labels-avoid]") ?? [])].map((e) => {
      const r = e.getBoundingClientRect();
      return { x0: r.left - box.left, x1: r.right - box.left, y0: r.top - box.top, y1: r.bottom - box.top };
    });
    let shown = 0;
    let waiting = false;
    const order = labels
      .map((l) => ({ l, d: eye.distanceTo(l.p) }))
      .sort((a, b) => +!a.l.place.major - +!b.l.place.major || +!a.l.on - +!b.l.on || a.d - b.d);
    for (const { l, d } of order) {
      let rect: Rect | null = null;
      let side = l.side;
      v.copy(l.p).applyMatrix4(worldFromMap).project(camera);
      const x = (v.x * 0.5 + 0.5) * width, y = (-v.y * 0.5 + 0.5) * height;
      const inView = v.z > -1 && v.z < 1 && x > -20 && x < width + 20 && y > 0 && y < height + 20;
      if (shown < MAX_SHOWN && inView && d < (l.place.major ? MAJOR_FAR_M : FAR_M)) {
        const near = d < NEAR_M;
        if (near !== l.near) l.el.classList.toggle("near", near);
        l.near = near;
        const level = near ? "near" : "far";
        const [w, h] = (l.dims[level] ??= [l.txt.offsetWidth || 160, l.txt.offsetHeight || 20]);
        const y0 = y - STEM - h, y1 = y + MARGIN;
        const rects: Record<Side, Rect> = { r: { x0: x - MARGIN, x1: x + w, y0, y1 }, l: { x0: x - w, x1: x + MARGIN, y0, y1 } };
        const fits = (r: Rect) => r.x0 >= MARGIN && r.x1 <= width - MARGIN && r.y0 >= MARGIN && !taken.some((t) => overlaps(r, t));
        const found = ([l.side, l.side === "r" ? "l" : "r"] as Side[]).find((s) => fits(rects[s]));
        if (found) {
          side = found;
          rect = rects[found];
        }
      }
      if (!rect) {
        l.since = -1;
        show(l, false);
        continue;
      }
      taken.push(rect); // also while waiting, so two labels don't settle on one spot
      if (l.since < 0) l.since = now;
      const on = l.on || now - l.since >= SETTLE_MS;
      waiting ||= !on;
      if (!on) continue;
      shown++;
      if (side !== l.side) l.el.classList.toggle("left", side === "l");
      l.side = side;
      l.el.style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
      show(l, true);
    }
    return waiting;
  };

  return {
    update,
    setLift: (metres) => {
      lift = metres;
      onChange();
    },
    dispose: () => {
      disposed = true;
      root.remove();
    },
  };
}
