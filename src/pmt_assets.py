"""Download a pinned, checksum-verified PMT terrain demonstration bundle."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import io
import tarfile
from pathlib import Path
import ssl
import urllib.request
from urllib.parse import quote

import certifi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/pmt'
REVISION = '38f9f7dba893385c6219620b72314396247c9e82'
BASE = f'https://huggingface.co/datasets/aCodeDog/PMT-assets/resolve/{REVISION}/'
MANIFEST_SHA = '73d3927346797c5bf371aaf3e4a86b851e9524ab05338a3167e75cc2f1eb457f'
TEACHER = 'checkpoints/pretrained/walkdance_bigmap_teacher.pt'
TERRAIN = 'assets/terrain/g1_29dof_big_map.stl'
URDF_SHA = 'dbc8471004e9b0b85c27a4ebc1f0d90a727c1cf7ca9ef507b805596f2004d64b'
ROBOT_ARCHIVE_SHA = 'b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674'

def robot_description():
    path = DATA / 'g1.urdf'
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == URDF_SHA:
        return path
    url = 'https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz'
    with urllib.request.urlopen(url, context=ssl.create_default_context(cafile=certifi.where()), timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != ROBOT_ARCHIVE_SHA:
        raise ValueError('Robot archive checksum mismatch')
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        source = archive.extractfile('unitree_description/urdf/g1/main.urdf')
        if source is None:
            raise ValueError('Robot description missing from archive')
        urdf = source.read()
    if hashlib.sha256(urdf).hexdigest() != URDF_SHA:
        raise ValueError('Robot description checksum mismatch')
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_bytes(urdf)
    return path

MOTIONS = 'assets/motions/terrain_mocaphouse/walk_dance1sub2start/optimized/'


def fetch(name, sha):
    path = DATA / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(BASE + quote(name, safe='/'), context=context, timeout=120) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != sha:
        raise ValueError(f'PMT checksum mismatch: {name}')
    temporary = path.with_suffix(path.suffix + '.download')
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def manifest():
    path = fetch('SHA256SUMS.txt', MANIFEST_SHA)
    entries = {}
    for line in path.read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        name = name.lstrip('*')
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe path in upstream manifest')
        entries[name] = sha
    return entries


def verify(name):
    entries = manifest()
    path = DATA / name
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entries[name]:
        raise ValueError(f'Missing or changed asset: {name}. Run just pmt-setup.')
    return path


def prepare():
    entries = manifest()
    selected = {k: v for k, v in entries.items()
                if k.startswith(MOTIONS) or k in (TEACHER, TERRAIN)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, name, sha) for name, sha in selected.items()]
        for i, future in enumerate(futures, 1):
            future.result()
            if i % 20 == 0:
                print(f'Checked/downloaded {i}/{len(futures)} PMT files', flush=True)
    report = {'source': 'https://huggingface.co/datasets/aCodeDog/PMT-assets',
              'revision': REVISION, 'files': selected,
              'purpose': 'Terrain reference data and teacher; physical transfer must be qualified separately.'}
    (DATA / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'PMT assets ready: {DATA}', flush=True)


if __name__ == '__main__':
    prepare()
