import * as THREE from "three";

/** A version's place on the map (worker.pipeline.georef), from its videos' GPS. */
export type Geo = {
  origin: [number, number];
  /** Row-major 4x4: map frame (metres east/north on the fitted ground plane) to scene. */
  scene_from_map: number[][];
  /** Camera centres' east/north bounds in metres: [minE, minN, maxE, maxN]. */
  extent: [number, number, number, number];
  error_m: number;
  /** Heights under the map (worker.pipeline.terrain): float32 map-frame z, north row first. */
  terrain?: Terrain;
};

export type Terrain = { url: string; size: [number, number]; min: [number, number]; spacing_m: number };

export type Heights = (x: number, y: number) => number;

/** Bilinear height at map-frame x, y; beyond the grid, its edge. */
function heightsOf(t: Terrain, z: Float32Array): Heights {
  const [nx, ny] = t.size;
  return (x, y) => {
    const c = Math.min(Math.max((x - t.min[0]) / t.spacing_m, 0), nx - 1.001);
    const r = Math.min(Math.max(ny - 1 - (y - t.min[1]) / t.spacing_m, 0), ny - 1.001);
    const j = Math.floor(c), i = Math.floor(r), fc = c - j, fr = r - i;
    const at = (ii: number, jj: number) => z[ii * nx + jj];
    return (at(i, j) * (1 - fc) + at(i, j + 1) * fc) * (1 - fr) + (at(i + 1, j) * (1 - fc) + at(i + 1, j + 1) * fc) * fr;
  };
}

// One fetch per terrain grid, shared by the map and the place labels.
const loadedHeights = new Map<string, Promise<Heights>>();

/** The terrain's height sampler, once its grid has loaded. */
export function loadHeights(t: Terrain): Promise<Heights> {
  let heights = loadedHeights.get(t.url);
  if (!heights) {
    heights = fetch(t.url)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
      .then((buffer) => heightsOf(t, new Float32Array(buffer)));
    heights.catch(() => loadedHeights.delete(t.url));
    loadedHeights.set(t.url, heights);
  }
  return heights;
}

export type MapStyle = "street" | "satellite";

type Provider = { name: string; attribution: string; maxZoom: number; url: (z: number, x: number, y: number) => string };

// Tried in order per tile: OSM first, then other street maps if its tiles don't load.
const PROVIDERS: Record<MapStyle, Provider[]> = {
  street: [
    { name: "OpenStreetMap", attribution: "© OpenStreetMap contributors", maxZoom: 19,
      url: (z, x, y) => `https://tile.openstreetmap.org/${z}/${x}/${y}.png` },
    { name: "CARTO", attribution: "© OpenStreetMap contributors © CARTO", maxZoom: 20,
      url: (z, x, y) => `https://basemaps.cartocdn.com/rastertiles/voyager/${z}/${x}/${y}.png` },
    { name: "Esri", attribution: "Tiles © Esri", maxZoom: 19,
      url: (z, x, y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/${z}/${y}/${x}` },
  ],
  satellite: [
    { name: "Esri World Imagery", attribution: "Imagery © Esri, Maxar, Earthstar Geographics", maxZoom: 19,
      url: (z, x, y) => `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}` },
  ],
};

const EARTH_CIRCUMFERENCE = 40075016.686;

/** Web-Mercator tile coordinates (fractional) of a lat/lon at zoom z. */
function tileXY(lat: number, lon: number, z: number): [number, number] {
  const n = 2 ** z;
  const phi = (lat * Math.PI) / 180;
  return [((lon + 180) / 360) * n, ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * n];
}

export type MapLayer = {
  group: THREE.Group;
  /** Load tiles around the camera (world position), wherever it has flown. */
  follow: (cameraWorld: THREE.Vector3) => void;
  /** Raise (or lower) the map by metres, to line it up with the scene by eye. */
  setLift: (metres: number) => void;
  dispose: () => void;
};

/** A quadtree of tiles out to RANGE_M around the camera: a tile is split into its four
 * children while the camera is closer than SPLIT of its sizes (counting the camera's height),
 * down to tiles about the camera's height above the map (TILE_MIN_M..TILE_MAX_M), so detail
 * falls off with distance and the tiles never overlap (overlapping levels flicker against
 * each other). The finest level changes once the height is ZOOM_HYSTERESIS levels past it,
 * so hovering near a boundary doesn't reload. Tiles that were replaced stay, pushed behind,
 * until their replacements have loaded (or RETIRE_MS), so the map doesn't blink on a change.
 * With a single level of RADIUS 4 tiles the map ended ~1 km away at a low flight. */
const RANGE_M = 8000;
const SPLIT = 2;
const TILE_MIN_M = 75;
const TILE_MAX_M = 5000;
const ZOOM_HYSTERESIS = 0.7;
const RETIRE_MS = 3000;
/** Terrain mesh: a vertex every grid step, at most this many quads along a tile's side. */
const MAX_SEGMENTS = 48;

/**
 * Map tiles as textured quads in the map frame (x east, y north, metres; tiles are spaced by
 * the local Mercator scale at the origin, exact enough over a few km), placed in the scene by
 * geo.scene_from_map, loaded around the camera as it flies (follow).
 */
export function buildMap(
  geo: Geo,
  style: MapStyle,
  onChange: () => void,
  onSource: (attribution: string, failed: boolean) => void,
): MapLayer {
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  const sceneFromMap = new THREE.Matrix4().set(...(geo.scene_from_map.flat() as Parameters<THREE.Matrix4["set"]>));
  group.matrix.copy(sceneFromMap);
  const [lat, lon] = geo.origin;
  const providers = PROVIDERS[style];
  const maxZoom = providers[0].maxZoom;
  const metresPerTile = (z: number) => (EARTH_CIRCUMFERENCE * Math.cos((lat * Math.PI) / 180)) / 2 ** z;

  const loader = new THREE.TextureLoader().setCrossOrigin("anonymous");
  type Tile = {
    mesh: THREE.Mesh; material: THREE.MeshBasicMaterial; geometry: THREE.PlaneGeometry; centre: THREE.Vector2;
    /** Its texture loaded, or no provider served it. */
    settled: boolean;
  };
  const tiles = new Map<string, Tile>();
  let disposed = false;
  let failures = 0;

  // Terrain: tiles made before it loads are draped when it does.
  let heights: Heights | null = null;
  const drape = (tile: Tile) => {
    if (!heights) return;
    const position = tile.geometry.attributes.position;
    for (let k = 0; k < position.count; k++)
      position.setZ(k, heights(tile.centre.x + position.getX(k), tile.centre.y + position.getY(k)));
    position.needsUpdate = true;
    tile.geometry.computeBoundingSphere();
  };
  if (geo.terrain) {
    loadHeights(geo.terrain).then((loaded) => {
      if (disposed) return;
      heights = loaded;
      tiles.forEach(drape);
      onChange();
    }).catch((e) => console.warn("terrain not loaded:", e));
  }

  const addTile = (z: number, tx: number, ty: number) => {
    const key = `${z}/${tx}/${ty}`;
    if (tiles.has(key)) return;
    const m = metresPerTile(z);
    const [ox, oy] = tileXY(lat, lon, z);
    const segments = geo.terrain ? Math.min(Math.max(Math.ceil(m / geo.terrain.spacing_m), 1), MAX_SEGMENTS) : 1;
    const geometry = new THREE.PlaneGeometry(m, m, segments, segments);
    const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set((tx + 0.5 - ox) * m, -(ty + 0.5 - oy) * m, 0);
    mesh.renderOrder = -2;
    mesh.visible = false;
    group.add(mesh);
    const tile: Tile = { mesh, material, geometry, centre: new THREE.Vector2(mesh.position.x, mesh.position.y), settled: false };
    tiles.set(key, tile);
    drape(tile);
    const attempt = (k: number) => {
      if (k >= providers.length) {
        tile.settled = true;
        onChange();
        if (++failures === 1) onSource(providers[0].attribution, true);
        return;
      }
      const provider = providers[k];
      loader.load(provider.url(Math.min(z, provider.maxZoom), tx, ty), (texture) => {
        if (disposed || !tiles.has(key)) return texture.dispose();
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = 8;
        material.map = texture;
        material.needsUpdate = true;
        mesh.visible = true;
        tile.settled = true;
        onSource(provider.attribution, false);
        onChange();
      }, undefined, () => !disposed && tiles.has(key) && attempt(k + 1));
    };
    attempt(0);
  };

  const removeTile = (key: string) => {
    const tile = tiles.get(key);
    if (!tile) return;
    tiles.delete(key);
    retiring.delete(key);
    tile.mesh.removeFromParent();
    tile.material.map?.dispose();
    tile.material.dispose();
    tile.geometry.dispose();
  };

  // Tiles replaced but kept until the new ones show, drawn behind them.
  const retiring = new Set<string>();
  let retiringSince = 0;
  const retire = (key: string, on: boolean) => {
    const tile = tiles.get(key)!;
    tile.material.polygonOffset = on;
    tile.material.polygonOffsetFactor = tile.material.polygonOffsetUnits = on ? 4 : 0;
    tile.material.needsUpdate = true;
    tile.mesh.renderOrder = on ? -3 : -2;
    if (on) retiring.add(key);
    else retiring.delete(key);
  };

  const local = new THREE.Vector3();
  const planned = new THREE.Vector3(Infinity, Infinity, Infinity);
  const inverse = new THREE.Matrix4();
  let zoom = 0;
  let wanted = new Set<string>();

  /** The quadtree's leaves around the camera (map frame) at `height` above the ground. */
  const plan = (height: number) => {
    const next = new Set<string>();
    const visit = (z: number, x: number, y: number) => {
      const m = metresPerTile(z);
      const [ox, oy] = tileXY(lat, lon, z);
      const x0 = (x - ox) * m, y1 = -(y - oy) * m;
      const flat = Math.hypot(Math.max(x0 - local.x, 0, local.x - x0 - m), Math.max(y1 - m - local.y, 0, local.y - y1));
      if (flat > RANGE_M) return;
      if (z < zoom && Math.hypot(flat, height) < SPLIT * m) {
        for (const [i, j] of [[0, 0], [1, 0], [0, 1], [1, 1]]) visit(z + 1, 2 * x + i, 2 * y + j);
      } else next.add(`${z}/${x}/${y}`);
    };
    // Roots: the level whose tiles are about RANGE_M across.
    const z0 = Math.min(Math.max(Math.floor(Math.log2(metresPerTile(0) / RANGE_M)), 0), zoom);
    const m = metresPerTile(z0);
    const [ox, oy] = tileXY(lat, lon, z0);
    for (let x = Math.floor(ox + (local.x - RANGE_M) / m); x <= Math.floor(ox + (local.x + RANGE_M) / m); x++)
      for (let y = Math.floor(oy - (local.y + RANGE_M) / m); y <= Math.floor(oy - (local.y - RANGE_M) / m); y++) visit(z0, x, y);
    return next;
  };

  const follow = (cameraWorld: THREE.Vector3) => {
    if (disposed) return;
    group.updateWorldMatrix(true, false);
    local.copy(cameraWorld).applyMatrix4(inverse.copy(group.matrixWorld).invert());
    const ground = heights ? heights(local.x, local.y) : 0;
    const height = Math.min(Math.max(Math.abs(local.z - ground), TILE_MIN_M), TILE_MAX_M);
    const finest = Math.min(Math.max(Math.log2(metresPerTile(0) / height), 3), maxZoom);
    let changed = false;
    if (!zoom || Math.abs(finest - zoom) > ZOOM_HYSTERESIS) {
      const next = Math.round(finest);
      changed = next !== zoom;
      zoom = next;
    }
    // Planned again once the camera has moved a quarter of a finest tile, or the level changed.
    if (changed || local.distanceTo(planned) > metresPerTile(zoom) / 4) {
      planned.copy(local);
      wanted = plan(height);
      for (const key of tiles.keys()) {
        if (wanted.has(key)) {
          if (retiring.has(key)) retire(key, false);
        } else if (!retiring.has(key)) {
          if (!tiles.get(key)!.settled) removeTile(key);
          else {
            retire(key, true);
            retiringSince = performance.now();
          }
        }
      }
      for (const key of wanted) {
        const [z, x, y] = key.split("/").map(Number);
        addTile(z, x, y);
      }
      onChange();
    }
    if (retiring.size && ([...wanted].every((k) => tiles.get(k)?.settled) || performance.now() - retiringSince > RETIRE_MS)) {
      for (const key of retiring) removeTile(key);
      retiring.clear();
      onChange();
    }
  };

  const setLift = (metres: number) => {
    group.matrix.multiplyMatrices(sceneFromMap, new THREE.Matrix4().makeTranslation(0, 0, metres));
    group.matrixWorldNeedsUpdate = true;
    onChange();
  };

  return {
    group,
    follow,
    setLift,
    dispose: () => {
      disposed = true;
      group.removeFromParent();
      [...tiles.keys()].forEach(removeTile);
    },
  };
}
