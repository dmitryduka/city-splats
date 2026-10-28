import * as THREE from "three";
import type { Geo } from "./mapTiles";

/** A splat training block (worker.pipeline.splat.blocks): its ground cells in the map frame
 * ([minE, minN, maxE, maxN], metres) and how far training has got with it. */
export type BlockOverlay = {
  key: string;
  cells: [number, number, number, number][];
  status: "done" | "training" | "queued";
  /** Position in training order (1-based) and the number of blocks. */
  index: number;
  count: number;
  psnr?: number | null;
};

export const BLOCK_COLORS: Record<BlockOverlay["status"], string> = {
  done: "#5cc87a",
  training: "#ffb340",
  queued: "#8fa3b8",
};

/** Cells are clipped to the flight plus this margin: blocks own all ground nearest to them,
 * so a block's cells can reach kilometres past where any frame looked. */
const MARGIN = 120;
/** Just above the map (z = 0), which lies half a metre under the fitted ground. */
const LIFT = 0.1;
/** Label height as a share of the view's height. */
const LABEL_HEIGHT = 0.035;

type Rect = [number, number, number, number];
type Segment = { axis: 0 | 1; at: number; from: number; to: number; side: -1 | 1 };

function clip(cell: Rect, area: Rect): Rect | null {
  const r: Rect = [Math.max(cell[0], area[0]), Math.max(cell[1], area[1]), Math.min(cell[2], area[2]), Math.min(cell[3], area[3])];
  return r[0] < r[2] && r[1] < r[3] ? r : null;
}

/** A rectangle's sides; `side` says which way the rectangle lies from the side. */
function sides([x0, y0, x1, y1]: Rect): Segment[] {
  return [
    { axis: 0, at: x0, from: y0, to: y1, side: 1 },
    { axis: 0, at: x1, from: y0, to: y1, side: -1 },
    { axis: 1, at: y0, from: x0, to: x1, side: 1 },
    { axis: 1, at: y1, from: x0, to: x1, side: -1 },
  ];
}

/** The block's outline: its cells' sides minus the parts shared with another of its cells
 * (a grouped block's cells are drawn as one area). */
function outline(cells: Rect[]): Segment[] {
  const all = cells.flatMap(sides);
  const out: Segment[] = [];
  for (const s of all) {
    let parts: [number, number][] = [[s.from, s.to]];
    for (const o of all) {
      if (o.axis !== s.axis || o.at !== s.at || o.side === s.side) continue;
      parts = parts.flatMap(([a, b]): [number, number][] => {
        if (o.to <= a || o.from >= b) return [[a, b]];
        return ([[a, Math.min(b, o.from)], [Math.max(a, o.to), b]] as [number, number][]).filter(([p, q]) => q > p);
      });
    }
    for (const [from, to] of parts) out.push({ ...s, from, to });
  }
  return out;
}

function label(text: string, color: string): THREE.Sprite {
  const canvas = document.createElement("canvas");
  const font = 28;
  const g = canvas.getContext("2d")!;
  g.font = `600 ${font}px system-ui, sans-serif`;
  canvas.width = Math.ceil(g.measureText(text).width) + 24;
  canvas.height = font + 16;
  g.font = `600 ${font}px system-ui, sans-serif`;
  g.fillStyle = "rgba(17, 19, 24, 0.75)";
  g.fillRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = color;
  g.textBaseline = "middle";
  g.fillText(text, 12, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true,
    sizeAttenuation: false }));
  sprite.userData.aspect = canvas.width / canvas.height;
  sprite.renderOrder = 10;
  return sprite;
}

export type BlockLayer = { group: THREE.Group; dispose: () => void };

/** Blocks as tinted squares with outlines on the map plane, each labelled with its order and
 * status. They are drawn after the map without depth testing (the map's polygon offset would
 * hide anything a few centimetres above it) and before the splats, so the scene stays on top. */
export function buildBlocks(geo: Geo, blocks: BlockOverlay[]): BlockLayer {
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  group.matrix.set(...(geo.scene_from_map.flat() as Parameters<THREE.Matrix4["set"]>));
  const [minE, minN, maxE, maxN] = geo.extent;
  const area: Rect = [minE - MARGIN, minN - MARGIN, maxE + MARGIN, maxN + MARGIN];
  const disposables: { dispose: () => void }[] = [];
  // Labels keep a fixed share of the screen height; the group's scale (scene units per metre)
  // would otherwise shrink them with it.
  const metre = new THREE.Vector3().setFromMatrixColumn(group.matrix, 0).length();
  const labelHeight = LABEL_HEIGHT / metre;

  for (const block of blocks) {
    const cells = block.cells.map((c) => clip(c, area)).filter((c): c is Rect => c !== null);
    if (!cells.length) continue;
    const color = new THREE.Color(BLOCK_COLORS[block.status]);

    const fill = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: block.status === "training" ? 0.28 : 0.14,
      depthTest: false, depthWrite: false, side: THREE.DoubleSide });
    disposables.push(fill);
    for (const [x0, y0, x1, y1] of cells) {
      const geometry = new THREE.PlaneGeometry(x1 - x0, y1 - y0);
      disposables.push(geometry);
      const mesh = new THREE.Mesh(geometry, fill);
      mesh.position.set((x0 + x1) / 2, (y0 + y1) / 2, LIFT);
      mesh.renderOrder = -0.5;
      group.add(mesh);
    }

    // Inset a little so neighbouring blocks' outlines don't overdraw each other.
    const inset = 0.6;
    const positions = outline(cells).flatMap(({ axis, at, from, to, side }) => {
      const a = at + side * inset;
      return axis === 0 ? [a, from, LIFT, a, to, LIFT] : [from, a, LIFT, to, a, LIFT];
    });
    const lines = new THREE.BufferGeometry();
    lines.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const lineMaterial = new THREE.LineBasicMaterial({ color, depthTest: false, depthWrite: false });
    disposables.push(lines, lineMaterial);
    const outlineMesh = new THREE.LineSegments(lines, lineMaterial);
    outlineMesh.renderOrder = -0.5;
    group.add(outlineMesh);

    // Label over the block's largest cell, sized to it.
    const big = cells.reduce((a, b) => ((b[2] - b[0]) * (b[3] - b[1]) > (a[2] - a[0]) * (a[3] - a[1]) ? b : a));
    const status = block.status === "done" ? (block.psnr != null ? `${block.psnr.toFixed(2)} dB` : "trained")
      : block.status;
    const sprite = label(`${block.index}/${block.count} ${block.key} · ${status}`, BLOCK_COLORS[block.status]);
    sprite.scale.set(labelHeight * sprite.userData.aspect, labelHeight, 1);
    sprite.position.set((big[0] + big[2]) / 2, (big[1] + big[3]) / 2, LIFT + 2);
    disposables.push(sprite.material.map!, sprite.material);
    group.add(sprite);
  }

  // Frustum culling hid all of them from views that clearly showed them (cause not found);
  // there are only a few dozen objects, so skip it.
  group.traverse((o) => (o.frustumCulled = false));
  return {
    group,
    dispose: () => {
      group.removeFromParent();
      disposables.forEach((d) => d.dispose());
    },
  };
}

type BlockResult = { cells?: Rect[]; eval?: { psnr?: number } | null };

/** A version's blocks with their status, from its metrics: the plan stored when training
 * started, the running report's finished blocks, or a finished run's blocks (of the viewed
 * splat kind if it has them). */
export function blockOverlays(metrics: Record<string, unknown>, training: boolean, kind: string | null): BlockOverlay[] {
  const runs = Object.entries(metrics)
    .filter(([k, m]) => k.startsWith("splat_") && (m as { blocks?: unknown })?.blocks)
    .sort(([a]) => (a === `splat_${kind}` ? -1 : 1));
  const finished = training ? null : ((runs[0]?.[1] as { blocks: Record<string, BlockResult> } | undefined)?.blocks ?? null);
  const report = (metrics.training_report as { blocks?: Record<string, BlockResult> } | undefined)?.blocks ?? {};
  const plan = (finished && Object.entries(finished).map(([key, b]) => ({ key, cells: b.cells ?? [] })))
    ?? (metrics.block_plan as { key: string; cells: Rect[] }[] | undefined) ?? [];
  let current = training;
  return plan.map(({ key, cells }, i) => {
    const result = finished?.[key] ?? report[key];
    let status: BlockOverlay["status"] = result ? "done" : "queued";
    if (!result && current) {
      status = "training";
      current = false;
    }
    return { key, cells, status, index: i + 1, count: plan.length, psnr: result?.eval?.psnr ?? null };
  });
}
