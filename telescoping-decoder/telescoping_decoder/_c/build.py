"""Compile and cache the S3 CPU kernels at runtime.

The package ships the ``.c`` sources and compiles them on first use with the
benchmarked flags::

    gcc -O3 -ffast-math -shared -fPIC -o <out>.so <src>.c -lm

``-march=native`` is disabled by default because a different architecture
under ``-ffast-math`` can change floating-point results. Enable it with
``TELESCOPING_DECODER_MARCH_NATIVE=1``.

Environment overrides take precedence over the cache and can select a
prebuilt binary::

    CHECKSERIAL_BP_SO=/path/to/checkserial_bp.so
    RELAY_MEM_BP_SO=/path/to/relay_mem_bp.so

Prebuild eagerly (e.g. in a container image) with::

    python -m telescoping_decoder.build
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_C_DIR = Path(__file__).resolve().parent
KERNELS = ("checkserial_bp", "relay_mem_bp")


def _compiler() -> str:
    """Resolve a C compiler: $CC, then PATH, then the Python env's bin dir.

    The last candidate covers conda environments that ship gcc without
    putting it on the caller's PATH.
    """
    exe_dir = Path(sys.executable).resolve().parent
    candidates = []
    if os.environ.get("CC"):
        candidates.append(os.environ["CC"])
    candidates += ["gcc", "cc", str(exe_dir / "gcc"), str(exe_dir / "cc")]
    for c in candidates:
        found = shutil.which(c)
        if found:
            return found
    raise RuntimeError(
        "no C compiler found (tried $CC, gcc, cc, and the Python env's bin "
        "dir); install gcc, or point CHECKSERIAL_BP_SO / RELAY_MEM_BP_SO at "
        "prebuilt libraries")


def _flags() -> list[str]:
    flags = ["-O3", "-ffast-math", "-shared", "-fPIC"]
    if os.environ.get("TELESCOPING_DECODER_MARCH_NATIVE") == "1":
        flags.insert(2, "-march=native")
    return flags


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = Path(base) / "telescoping_decoder"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_lib(name: str) -> str:
    """Return the path to the compiled ``<name>.so``, building if needed.

    Precedence: ``<NAME>_SO`` env override > cached build keyed on
    sha256(source + flags) > fresh gcc compile (atomic rename, so concurrent
    processes race safely).
    """
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; expected one of {KERNELS}")

    env_override = os.environ.get(f"{name.upper()}_SO")
    if env_override:
        if not os.path.isfile(env_override):
            raise FileNotFoundError(
                f"{name.upper()}_SO={env_override} does not exist")
        return env_override

    src = _C_DIR / f"{name}.c"
    source = src.read_bytes()
    flags = _flags()
    cc = _compiler()
    key = hashlib.sha256(
        source + "\0".join([cc, *flags]).encode()).hexdigest()[:16]
    out = _cache_dir() / f"{name}-{key}.so"
    if out.is_file():
        return str(out)

    fd, tmp = tempfile.mkstemp(suffix=".so", dir=str(out.parent))
    os.close(fd)
    try:
        cmd = [cc, *flags, "-o", tmp, str(src), "-lm"]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"{cc} failed compiling {src.name}:\n{e.stderr}") from e
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(out)


def main() -> None:
    for name in KERNELS:
        print(f"{name}: {ensure_lib(name)}")


if __name__ == "__main__":
    main()
