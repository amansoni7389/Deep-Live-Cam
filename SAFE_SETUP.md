# Safer CPU-only setup

This mode is intended for an isolated test VM. It disables camera discovery,
the random-face download, model downloads, mirror downloads, and external
website links. It does not make face-swapping safe for real calls or remove
the need for consent.

The reviewed source revision is
`08896a51c9a6198f50dc9f7508acf39fc460c8f1`. Before installing, verify it:

```bash
test "$(git rev-parse HEAD)" = \
  "08896a51c9a6198f50dc9f7508acf39fc460c8f1"
git status --short
```

On Ubuntu 24.04 with Python 3.12:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ffmpeg libegl1 libxcb-cursor0 libxkbcommon-x11-0 \
  python3.12-dev python3.12-venv

python3.12 -m venv venv-safe
venv-safe/bin/python -m pip install \
  pip==25.3 setuptools==80.9.0 wheel==0.45.1 \
  numpy==2.2.6 Cython==3.1.4

curl --fail --location \
  https://files.pythonhosted.org/packages/0b/8d/0f4af90999ca96cf8cb846eb5ae27c5ef5b390f9c090dd19e4fa76364c13/insightface-0.7.3.tar.gz \
  --output /tmp/insightface-0.7.3.tar.gz
echo \
  "f191f719612ebb37018f41936814500544cd0f86e6fcd676c023f354c668ddf7  /tmp/insightface-0.7.3.tar.gz" \
  | sha256sum --check -
venv-safe/bin/python -m pip wheel --no-build-isolation --no-deps \
  /tmp/insightface-0.7.3.tar.gz --wheel-dir /tmp/dlc-wheelhouse
venv-safe/bin/python -m pip install \
  /tmp/dlc-wheelhouse/insightface-0.7.3-*.whl

venv-safe/bin/python -m pip install -r requirements-safe-cpu.txt
venv-safe/bin/python -m pip check
```

The InsightFace package is distributed as source for CPython 3.12 and builds
a local Cython/C++ extension during installation. Its expected PyPI sdist
SHA-256 is:

```text
f191f719612ebb37018f41936814500544cd0f86e6fcd676c023f354c668ddf7
```

Provision models once while network access is restricted to Hugging Face.
The application pins repository revision
`966b5ea2adc8091d0230c579a09ed541c711de80` and verifies each known model's
size and SHA-256 before loading it. Then disconnect the VM and launch:

```bash
DLC_OFFLINE_SAFE_MODE=1 venv-safe/bin/python run.py
```

Deep-Live-Cam is a native Qt application; it does not provide an HTTP URL.
Use only generated synthetic assets in this mode. Do not attach a webcam,
join a call, mount personal directories, or enable clipboard sharing.
