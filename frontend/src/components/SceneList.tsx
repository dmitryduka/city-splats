import { useState } from "react";
import { api, type Scene } from "../api";

type Props = {
  scenes: Scene[];
  current: string | null;
  onSelect: (slug: string) => void;
  onCreated: (scene: Scene) => void;
};

export function SceneList({ scenes, current, onSelect, onCreated }: Props) {
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) return;
    try {
      const scene = await api.createScene(name.trim());
      setName("");
      setError(null);
      onCreated(scene);
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <nav className="sidebar">
      <h2>Scenes</h2>
      <ul className="scene-list">
        {scenes.map((scene) => {
          const processing = (scene.counts?.processing ?? 0) + (scene.counts?.queued ?? 0);
          return (
            <li key={scene.slug}>
              <button className={scene.slug === current ? "active" : ""} onClick={() => onSelect(scene.slug)}>
                <span>{scene.name}</span>
                <span className="muted">
                  {scene.counts?.done ?? 0} videos{processing ? ` · ${processing} processing` : ""}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
      <form onSubmit={create} className="new-scene">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="New scene, e.g. Small street" />
        <button type="submit" disabled={!name.trim()}>
          Create
        </button>
      </form>
      {error && <div className="error">{error}</div>}
    </nav>
  );
}
