import { useEffect, useState } from "react";
import type { CameraReplay } from "../viewer/cameraPath";

export const REPLAY_SPEEDS = [1, 2, 4, 8];

const clock = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

/** Time, pause, seek, speed and stop for a running replay. The replay's clock advances in the
 * viewer's render loop, so this polls it rather than re-rendering the app every frame. */
export function ReplayBar({
  replay,
  label,
  onSpeed,
  onStop,
}: {
  replay: CameraReplay;
  label: string;
  onSpeed: (speed: number) => void;
  onStop: () => void;
}) {
  const [, setTick] = useState(0);
  const refresh = () => setTick((n) => n + 1);
  useEffect(() => {
    const timer = setInterval(refresh, 200);
    return () => clearInterval(timer);
  }, []);
  return (
    <div className="replay-bar">
      <button onClick={() => ((replay.paused = !replay.paused), refresh())} title="P">
        {replay.paused ? "▶" : "❚❚"}
      </button>
      <span className="replay-label" title={label}>{label}</span>
      <input
        type="range"
        min={0}
        max={replay.duration}
        step={0.05}
        value={replay.t}
        onChange={(e) => ((replay.t = Number(e.target.value)), refresh())}
      />
      <span className="replay-time">
        {clock(replay.t)} / {clock(replay.duration)}
      </span>
      <span className="surface-switch">
        {REPLAY_SPEEDS.map((s) => (
          <button key={s} className={replay.speed === s ? "on" : ""} onClick={() => onSpeed(s)}>
            {s}×
          </button>
        ))}
      </span>
      <button onClick={onStop} title="Esc">
        Stop
      </button>
    </div>
  );
}
