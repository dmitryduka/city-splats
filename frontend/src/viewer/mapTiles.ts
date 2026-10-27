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
/** Detail tiles: the finest zoom covering the flight (plus MARGIN metres) in at most MAX_TILES. */
const MAX_TILES = 64;
const MARGIN = 120;
/** Context: CONTEXT_ZOOM_OUT levels coarser, CONTEXT_RADIUS tiles around the centre. */
const CONTEXT_ZOOM_OUT = 3;
const CONTEXT_RADIUS = 2;

/** Web-Mercator tile coordinates (fractional) of a lat/lon at zoom z. */
function tileXY(lat: number, lon: number, z: number): [number, number] {
  const n = 2 ** z;
  const phi = (lat * Math.PI) / 180;
  return [((lon + 180) / 360) * n, ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * n];
}

export type MapLayer = { group: THREE.Group; dispose: () => void };

/**
 * Map tiles as textured quads in the map frame (x east, y north, metres; tiles are spaced by
 * the local Mercator scale at the origin, exact enough over a few km), placed in the scene by
 * geo.scene_from_map. A coarse ring of context tiles lies under the detailed ones.
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

  const [minE, minN, maxE, maxN] = geo.extent;
  let zoom = maxZoom;
  const span = (z: number) => {
    const m = metresPerTile(z);
    return (Math.floor((maxE + MARGIN) / m) - Math.floor((minE - MARGIN) / m) + 2) *
      (Math.floor((maxN + MARGIN) / m) - Math.floor((minN - MARGIN) / m) + 2);
  };
  while (zoom > 3 && span(zoom) > MAX_TILES) zoom--;

  const loader = new THREE.TextureLoader().setCrossOrigin("anonymous");
  const materials: THREE.MeshBasicMaterial[] = [];
  const geometries: THREE.PlaneGeometry[] = [];
  let disposed = false;
  let failures = 0;

  const addTile = (z: number, tx: number, ty: number, detail: boolean) => {
    const m = metresPerTile(z);
    const [ox, oy] = tileXY(lat, lon, z);
    const geometry = new THREE.PlaneGeometry(m, m);
    const material = new THREE.MeshBasicMaterial({ side: THREE.DoubleSide, polygonOffset: detail,
      polygonOffsetFactor: detail ? -1 : 0, polygonOffsetUnits: detail ? -4 : 0 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set((tx + 0.5 - ox) * m, -(ty + 0.5 - oy) * m, 0);
    mesh.renderOrder = detail ? -1 : -2;
    mesh.visible = false;
    group.add(mesh);
    materials.push(material);
    geometries.push(geometry);
    const attempt = (k: number) => {
      if (k >= providers.length) {
        if (++failures === 1) onSource(providers[0].attribution, true);
        return;
      }
      const provider = providers[k];
      loader.load(provider.url(Math.min(z, provider.maxZoom), tx, ty), (texture) => {
        if (disposed) return texture.dispose();
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = 8;
        material.map = texture;
        material.needsUpdate = true;
        mesh.visible = true;
        onSource(provider.attribution, false);
        onChange();
      }, undefined, () => !disposed && attempt(k + 1));
    };
    attempt(0);
  };

  // Detail tiles over the flight plus a margin.
  const m = metresPerTile(zoom);
  const [ox, oy] = tileXY(lat, lon, zoom);
  for (let tx = Math.floor(ox + (minE - MARGIN) / m); tx <= Math.floor(ox + (maxE + MARGIN) / m); tx++)
    for (let ty = Math.floor(oy - (maxN + MARGIN) / m); ty <= Math.floor(oy - (minN - MARGIN) / m); ty++)
      addTile(zoom, tx, ty, true);

  // Context: a ring of coarser tiles around the flight's centre.
  const coarse = Math.max(zoom - CONTEXT_ZOOM_OUT, 1);
  const mc = metresPerTile(coarse);
  const [cx, cy] = tileXY(lat, lon, coarse);
  const mx = Math.floor(cx + (minE + maxE) / 2 / mc);
  const my = Math.floor(cy - (minN + maxN) / 2 / mc);
  for (let tx = mx - CONTEXT_RADIUS; tx <= mx + CONTEXT_RADIUS; tx++)
    for (let ty = my - CONTEXT_RADIUS; ty <= my + CONTEXT_RADIUS; ty++) addTile(coarse, tx, ty, false);

  return {
    group,
    dispose: () => {
      disposed = true;
      group.removeFromParent();
      materials.forEach((m) => { m.map?.dispose(); m.dispose(); });
      geometries.forEach((g) => g.dispose());
    },
  };
}
