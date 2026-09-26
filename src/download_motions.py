"""Download the pinned full G1 source library; retain existing valid recordings."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def validate(path: Path) -> None:
    with np.load(path, allow_pickle=False) as clip:
        joints = clip['joint_pos']
        n = len(joints)
        if n < 1 or joints.shape != (n, 29):
            raise ValueError('Expected nonempty 29-joint motion')
        fps = np.asarray(clip['fps']).reshape(-1)
        if fps.size != 1 or not np.isfinite(fps).all() or fps[0] <= 0:
            raise ValueError('Invalid frame rate')
        shapes = {'joint_pos': (n, 29), 'body_pos_w': (n, 30, 3),
                  'body_quat_w': (n, 30, 4), 'body_lin_vel_w': (n, 30, 3),
                  'body_ang_vel_w': (n, 30, 3)}
        for key, shape in shapes.items():
            if clip[key].shape != shape or not np.isfinite(clip[key]).all():
                raise ValueError(f'Invalid {key}')
        if np.any(np.linalg.norm(clip['body_quat_w'], axis=-1) < 1e-6):
            raise ValueError('Zero body quaternion')


def download(item: dict, catalog: dict, directory: Path) -> str:
    destination = directory / (item['id'] + '.npz')
    if destination.exists():
        try:
            validate(destination)
            return f"Kept {item['id']}"
        except (OSError, ValueError, KeyError):
            pass
    url = f"https://huggingface.co/datasets/{catalog['dataset']}/resolve/{catalog['revision']}/{item['path']}"
    for attempt in range(3):
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            validate(temporary)
            temporary.replace(destination)
            return f"Downloaded {item['id']}"
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            temporary.unlink(missing_ok=True)
    raise RuntimeError('Download failed')


def main() -> None:
    catalog = json.loads((ROOT / 'src/motion_catalog.json').read_text())
    directory = ROOT / 'data/motions'
    directory.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda item: download(item, catalog, directory), catalog['clips']):
            print(result, flush=True)
    print(f"Ready: {len(catalog['clips'])} source clips. Physics tracking quality varies by clip.")


if __name__ == '__main__':
    main()
