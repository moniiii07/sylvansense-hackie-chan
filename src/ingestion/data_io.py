import json, hashlib, os
from datetime import datetime, timezone
import numpy as np

def save_scene(scene, bbox, start, end, aoi_name="aoi", base_dir="data/raw"):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    pull_id = f"{ts}_{aoi_name}"
    pull_dir = os.path.join(base_dir, pull_id)
    os.makedirs(pull_dir, exist_ok=True)

    np.savez_compressed(os.path.join(pull_dir, "bands.npz"), **scene.bands)

    band_bytes = b"".join(scene.bands[k].tobytes() for k in sorted(scene.bands))
    data_hash = hashlib.sha256(band_bytes).hexdigest()

    metadata = dict(
        pull_id=pull_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        bbox=bbox,
        date_range=[start, end],
        band_keys=sorted(scene.bands.keys()),
        shape=scene.shape,
        crs=scene.crs,
        data_sha256=data_hash,
    )
    with open(os.path.join(pull_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    _append_to_manifest(metadata, pull_dir)
    print(f"Saved to {pull_dir}")
    return pull_dir


def _append_to_manifest(metadata, pull_dir, manifest_path="data/manifest.json"):
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    manifest[metadata["pull_id"]] = {**metadata, "path": pull_dir}
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_scene(pull_dir):
    data = np.load(os.path.join(pull_dir, "bands.npz"))
    bands = {k: data[k] for k in data.files}
    with open(os.path.join(pull_dir, "metadata.json")) as f:
        meta = json.load(f)
    return bands, meta