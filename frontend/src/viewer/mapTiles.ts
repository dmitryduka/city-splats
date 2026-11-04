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

/** One zoom level at a time (overlapping levels flicker against each other): tiles about the
 * camera's height above the map (TILE_MIN_M..TILE_MAX_M), RADIUS tiles each way around the
 * point under it. The level changes once the height is ZOOM_HYSTERESIS levels past the
 * current one, so hovering near a boundary doesn't reload. Beyond MAX_LOADED tiles the ones
 * farthest from the camera go. */
const RADIUS = 4;
const TILE_MIN_M = 75;
const TILE_MAX_M = 5000;
const ZOOM_HYSTERESIS = 0.7;
const MAX_LOADED = 200;
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
  type Tile = { mesh: THREE.Mesh; material: THREE.MeshBasicMaterial; geometry: THREE.PlaneGeometry; centre: THREE.Vector2 };
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
    const tile = { mesh, material, geometry, centre: new THREE.Vector2(mesh.position.x, mesh.position.y) };
    tiles.set(key, tile);
    drape(tile);
    const attempt = (k: number) => {
      if (k >= providers.length) {
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
    tile.mesh.removeFromParent();
    tile.material.map?.dispose();
    tile.material.dispose();
    tile.geometry.dispose();
  };

  const local = new THREE.Vector3();
  const inverse = new THREE.Matrix4();
  let zoom = 0;
  let last = "";
  const follow = (cameraWorld: THREE.Vector3) => {
    if (disposed) return;
    group.updateWorldMatrix(true, false);
    local.copy(cameraWorld).applyMatrix4(inverse.copy(group.matrixWorld).invert());
    const ground = heights ? heights(local.x, local.y) : 0;
    const height = Math.min(Math.max(Math.abs(local.z - ground), TILE_MIN_M), TILE_MAX_M);
    const wanted = Math.min(Math.max(Math.log2(metresPerTile(0) / height), 3), maxZoom);
    if (!zoom || Math.abs(wanted - zoom) > ZOOM_HYSTERESIS) {
      const next = Math.round(wanted);
      if (next !== zoom) {
        for (const key of [...tiles.keys()]) removeTile(key);
        zoom = next;
        onChange();
      }
    }
    const m = metresPerTile(zoom);
    const [ox, oy] = tileXY(lat, lon, zoom);
    const mx = Math.floor(ox + local.x / m);
    const my = Math.floor(oy - local.y / m);
    // Only when the camera reaches another tile or level.
    const key = `${zoom}/${mx}/${my}`;
    if (key === last) return;
    last = key;
    for (let tx = mx - RADIUS; tx <= mx + RADIUS; tx++)
      for (let ty = my - RADIUS; ty <= my + RADIUS; ty++) addTile(zoom, tx, ty);
    if (tiles.size > MAX_LOADED) {
      const here = new THREE.Vector2(local.x, local.y);
      const far = [...tiles].sort(([, a], [, b]) => b.centre.distanceTo(here) - a.centre.distanceTo(here));
      for (const [k] of far.slice(0, tiles.size - MAX_LOADED)) removeTile(k);
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
