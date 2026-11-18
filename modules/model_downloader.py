import os
import hashlib
import ssl
import threading
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import certifi
from tqdm import tqdm

from modules.paths import MODELS_DIR

HF_REPO_ID = "hacksider/deep-live-cam"
# Immutable Hugging Face repository revision reviewed on 2026-11-18.
HF_REPO_REVISION = "966b5ea2adc8091d0230c579a09ed541c711de80"
HF_RESOLVE_BASE = (
    f"https://huggingface.co/{HF_REPO_ID}/resolve/{HF_REPO_REVISION}/"
)

MODEL_SIZES: Dict[str, int] = {
    "inswapper_128.onnx": 554253681,
    "inswapper_128_fp16.onnx": 277680638,
    "gfpgan-1024.onnx": 365875079,
    "GPEN-BFR-256.onnx": 75715262,
    "GPEN-BFR-512.onnx": 284244491,
    "buffalo_l/buffalo_l/1k3d68.onnx": 143607619,
    "buffalo_l/buffalo_l/2d106det.onnx": 5030888,
    "buffalo_l/buffalo_l/det_10g.onnx": 16923827,
    "buffalo_l/buffalo_l/genderage.onnx": 1322532,
    "buffalo_l/buffalo_l/w600k_r50.onnx": 174383860,
}

MODEL_SHA256: Dict[str, str] = {
    "inswapper_128.onnx": "e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    "inswapper_128_fp16.onnx": "6d51a9278a1f650cffefc18ba53f38bf2769bf4bbff89267822cf72945f8a38b",
    "gfpgan-1024.onnx": "ee8dd6415e388b3a410689d5d9395a2bf50b5973b588421ebfa57bc266f19e24",
    "GPEN-BFR-256.onnx": "aa5bd3ab238640a378c59e4a560f7a7150627944cf2129e6311ae4720e833271",
    "GPEN-BFR-512.onnx": "bf80acb8e91ba8852e3f012505be2c3b6cd6b3eed5ec605e3db87863c4e74d4e",
    "buffalo_l/buffalo_l/1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",
    "buffalo_l/buffalo_l/2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",
    "buffalo_l/buffalo_l/det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "buffalo_l/buffalo_l/genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",
    "buffalo_l/buffalo_l/w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}

_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()

CHUNK_SIZE = 1024 * 256


def _ssl_context() -> ssl.SSLContext:
    """TLS context for model downloads.

    Always verifies certificates against the certifi CA bundle, on every
    platform. Previously macOS used an unverified context, which let a
    network-position attacker substitute model files in transit (#1890).
    """
    return ssl.create_default_context(cafile=certifi.where())


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


def resolve_url(name: str) -> str:
    return HF_RESOLVE_BASE + name.replace(os.sep, "/")


def local_path(name: str, dest_dir: Optional[str] = None) -> str:
    if dest_dir is not None:
        return os.path.join(dest_dir, os.path.basename(name))
    return os.path.join(MODELS_DIR, *name.replace("/", os.sep).split(os.sep))


def expected_size(name: str) -> Optional[int]:
    return MODEL_SIZES.get(name.replace(os.sep, "/"))


def expected_sha256(name: str) -> Optional[str]:
    return MODEL_SHA256.get(name.replace(os.sep, "/"))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def is_present(name: str, dest_dir: Optional[str] = None) -> bool:
    path = local_path(name, dest_dir)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    size = expected_size(name)
    if size is not None and os.path.getsize(path) != size:
        return False
    digest = expected_sha256(name)
    return digest is None or sha256_file(path) == digest


def _download(name: str, url: str, target: str, size: Optional[int]) -> bool:
    os.makedirs(os.path.dirname(target) or MODELS_DIR, exist_ok=True)
    partial = target + ".part"
    resume_from = os.path.getsize(partial) if os.path.isfile(partial) else 0

    headers = {"User-Agent": "Deep-Live-Cam"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"

    try:
        request = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(request, context=_ssl_context(), timeout=60)
    except urllib.error.HTTPError as error:
        if resume_from and error.code in (416, 501):
            try:
                os.remove(partial)
            except OSError:
                pass
            return _download(name, url, target, size)
        print(f"[DLC.MODELS] Failed to download {name}: HTTP {error.code}")
        return False
    except (urllib.error.URLError, OSError) as error:
        print(f"[DLC.MODELS] Failed to download {name}: {error}")
        return False

    with response:
        if resume_from and getattr(response, "status", 200) != 206:
            resume_from = 0
        remaining = int(response.headers.get("Content-Length", 0) or 0)
        total = size or (resume_from + remaining) or None
        mode = "ab" if resume_from else "wb"
        try:
            with open(partial, mode) as handle:
                with tqdm(
                    total=total,
                    initial=resume_from,
                    desc=f"Downloading {os.path.basename(name)}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as progress:
                    while True:
                        buffer = response.read(CHUNK_SIZE)
                        if not buffer:
                            break
                        handle.write(buffer)
                        progress.update(len(buffer))
        except (urllib.error.URLError, OSError) as error:
            print(f"[DLC.MODELS] Download of {name} interrupted: {error}")
            return False

    downloaded = os.path.getsize(partial)
    if size is not None and downloaded != size:
        print(f"[DLC.MODELS] {name} is {downloaded} bytes, expected {size}. Discarding.")
        try:
            os.remove(partial)
        except OSError:
            pass
        return False

    digest = expected_sha256(name)
    if digest is not None:
        actual_digest = sha256_file(partial)
        if actual_digest != digest:
            print(
                f"[DLC.MODELS] {name} SHA-256 is {actual_digest}, "
                f"expected {digest}. Discarding."
            )
            try:
                os.remove(partial)
            except OSError:
                pass
            return False

    try:
        os.replace(partial, target)
    except OSError as error:
        print(f"[DLC.MODELS] Could not finalise {name}: {error}")
        return False
    return True


def ensure_model(
    name: str, quiet: bool = False, dest_dir: Optional[str] = None
) -> Optional[str]:
    name = name.replace(os.sep, "/")
    target = local_path(name, dest_dir)

    with _lock_for(target):
        if is_present(name, dest_dir):
            return target
        if os.environ.get("DLC_OFFLINE_SAFE_MODE") == "1":
            if not quiet:
                print(
                    f"[DLC.MODELS] {name} is missing; automatic downloads are "
                    "disabled by offline safe mode."
                )
            return None
        if not quiet:
            print(f"[DLC.MODELS] {name} not found in models folder, downloading...")
        if _download(name, resolve_url(name), target, expected_size(name)):
            return target
    return None


def ensure_any(names: List[str]) -> Optional[str]:
    for name in names:
        if is_present(name):
            return local_path(name)
    for name in names:
        path = ensure_model(name)
        if path is not None:
            return path
    return None


def ensure_insightface_pack(name: str = "buffalo_l") -> bool:
    members = [n for n in MODEL_SIZES if n.startswith(f"{name}/")]
    if not members:
        return False

    dest_dir = os.path.join(os.path.expanduser("~"), ".insightface", "models", name)
    if all(is_present(member, dest_dir) for member in members):
        return True

    print(f"[DLC.MODELS] insightface pack '{name}' is missing, downloading...")
    ok = True
    for member in members:
        if ensure_model(member, quiet=True, dest_dir=dest_dir) is None:
            ok = False
    if not ok:
        print(f"[DLC.MODELS] Could not pre-fill '{name}'; insightface will retry.")
    return ok
