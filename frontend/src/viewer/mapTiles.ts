import * as THREE from "three";

/** A version's place on the map (worker.pipeline.georef), from its videos' GPS. */
export type Geo = {
  origin: [number, number];
  /** Row-major 4x4: map frame (metres east/north on the fitted ground plane) to scene. */
  scene_from_map: number[][];
  /** Camera centres' east/north bounds in metres: [minE, minN, maxE, maxN]. */
  extent: [number, number, number, number];
  error_m: number;
};

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
  group.matrix.set(...(geo.scene_from_map.flat() as Parameters<THREE.Matrix4["set"]>));
  const [lat, lon] = geo.origin;
  const providers = PROVIDERS[style];
  const maxZoom = providers[0].maxZoom;
  const metresPerTile = (z: number) => (EARTH_CIRCUMFERENCE * Math.cos((lat * Math.PI) / 180)) / 2 ** z;

  const loader = new THREE.TextureLoader().setCrossOrigin("anonymous");
  type Tile = { mesh: THREE.Mesh; material: THREE.MeshBasicMaterial; geometry: THREE.PlaneGeometry; centre: THREE.Vector2 };
  const tiles = new Map<string, Tile>();
  let disposed = false;
  let failures = 0;

  const addTile = (z: number, tx: number, ty: number) => {
    const key = `${z}/${tx}/${ty}`;
    if (tiles.has(key)) return;
    const m = metresPerTile(z);
    const [ox, oy] = tileXY(lat, lon, z);
    const geometry = new THREE.PlaneGeometry(m, m);
    const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set((tx + 0.5 - ox) * m, -(ty + 0.5 - oy) * m, 0);
    mesh.renderOrder = -2;
    mesh.visible = false;
    group.add(mesh);
    tiles.set(key, { mesh, material, geometry, centre: new THREE.Vector2(mesh.position.x, mesh.position.y) });
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
    const height = Math.min(Math.max(Math.abs(local.z), TILE_MIN_M), TILE_MAX_M);
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

  return {
    group,
    follow,
    dispose: () => {
      disposed = true;
      group.removeFromParent();
      [...tiles.keys()].forEach(removeTile);
    },
  };
}
