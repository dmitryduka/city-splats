import * as THREE from "three";

/** camera_paths.json (worker.pipeline.camera_paths): per video (keyed by image prefix
 * "c<id>"), [t, x, y, z, qx, qy, qz, qw] rows sorted by t, the camera centre and
 * world_from_cam rotation in the reconstruction's frame with COLMAP camera axes; and under
 * "fov" each video's vertical field of view in degrees (missing in files written before it). */
export type CameraPaths = { [prefix: string]: number[][] } & { fov?: Record<string, number> };

/** The image prefix of a contribution's frames: its upload is <scene>/uploads/<id>_<file>,
 * and a cloned scene keeps the original's uploads (worker.runner._videos). */
export const videoPrefix = (videoPath: string) => "c" + videoPath.split("/").pop()!.split("_", 1)[0];

// A three.js camera looks down -z with y up; a COLMAP camera down +z with y down.
const COLMAP_FROM_THREE_CAMERA = new THREE.Quaternion(1, 0, 0, 0);

/**
 * Replays a captured camera path at the footage's pace (times `speed`): positions on a
 * centripetal Catmull-Rom through the keyframe centres (no overshoot or loops where keyframes
 * bunch up, as they do when the camera slows down), orientations slerped between keyframes.
 */
export class CameraReplay {
  readonly duration: number;
  t = 0;
  speed = 1;
  paused = false;
  /** The footage's vertical field of view (degrees), if known. */
  readonly fov: number | null;
  private times: number[];
  private curve: THREE.CatmullRomCurve3;
  private quats: THREE.Quaternion[];

  constructor(rows: number[][], fov: number | null = null) {
    this.fov = fov;
    // Keyframes sharing a timestamp would make a zero-length segment.
    const kept = rows.filter((r, i) => i === 0 || r[0] > rows[i - 1][0]);
    const t0 = kept[0][0];
    this.times = kept.map((r) => r[0] - t0);
    this.duration = this.times[this.times.length - 1];
    this.curve = new THREE.CatmullRomCurve3(kept.map((r) => new THREE.Vector3(r[1], r[2], r[3])), false, "centripetal");
    this.quats = kept.map((r) => new THREE.Quaternion(r[4], r[5], r[6], r[7]).multiply(COLMAP_FROM_THREE_CAMERA));
  }

  /** Advances the clock by `dt` real seconds; false once the end is reached. */
  step(dt: number): boolean {
    if (!this.paused) this.t = Math.min(this.duration, this.t + dt * this.speed);
    return this.t < this.duration;
  }

  /** The pose at the current time, in the reconstruction's frame, as a three.js camera's. */
  pose(position: THREE.Vector3, quaternion: THREE.Quaternion): void {
    const { times } = this;
    let lo = 0;
    let hi = times.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= this.t) lo = mid;
      else hi = mid;
    }
    const f = Math.min(1, Math.max(0, (this.t - times[lo]) / (times[hi] - times[lo])));
    // The curve's parameter is uniform per keyframe; map time into it segment by segment.
    this.curve.getPoint((lo + f) / (times.length - 1), position);
    quaternion.slerpQuaternions(this.quats[lo], this.quats[hi], f);
  }
}
