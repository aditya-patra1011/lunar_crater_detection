"""Download the raw datasets into ``data/raw``.

Each source is a small set of files with a known size and, once pinned, a SHA-256
checksum. Files stream to a ``.part`` file, resume after an interruption when the
server supports range requests, and are only renamed into place once verified.

Usage::

    python -m lunar_crater.data.download --list
    python -m lunar_crater.data.download robbins_2018 lola_ldem_64
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

CHUNK_SIZE = 1 << 20  # 1 MiB
DEFAULT_DEST = Path("data/raw")
# USGS answers 403 to urllib's default "Python-urllib" user agent.
USER_AGENT = "lunar-crater-detection (+https://github.com/aditya-patra1011/lunar_crater_detection)"


@dataclass(frozen=True)
class RemoteFile:
    url: str
    filename: str
    size: int
    sha256: str | None = None


@dataclass(frozen=True)
class Source:
    name: str
    description: str
    files: tuple[RemoteFile, ...]

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


_PDS_LOLA_GDR = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/"
    "lrolol_1xxx/data/lola_gdr/cylindrical/img"
)

SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            name="robbins_2018",
            description="Robbins (2019) lunar crater database, ~1.3M craters >= 1-2 km (USGS)",
            files=(
                RemoteFile(
                    url=(
                        "https://astrogeology.usgs.gov/ckan/dataset/"
                        "f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
                        "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/"
                        "lunar_crater_database_robbins_2018"
                    ),
                    filename="lunar_crater_database_robbins_2018.zip",
                    size=96_227_201,
                    sha256="694137e141138892555d0f976446875331d3317ea506ddf055d91b917ca307ab",
                ),
            ),
        ),
        Source(
            name="lola_ldem_118m",
            description="LOLA global DEM, 118 m/px (256 px/deg), BigTIFF (USGS)",
            files=(
                RemoteFile(
                    url=(
                        "https://planetarymaps.usgs.gov/mosaic/"
                        "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
                    ),
                    filename="Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif",
                    size=8_494_203_833,
                    sha256="de6fb1cfcb09eaa602918560224c5842b2eadcd159c59e8798fab6df00d7d9a0",
                ),
            ),
        ),
        Source(
            name="lola_ldem_64",
            description="LOLA global DEM, 474 m/px (64 px/deg), PDS image + label; for quick work",
            files=(
                RemoteFile(
                    url=f"{_PDS_LOLA_GDR}/ldem_64.img",
                    filename="ldem_64.img",
                    size=530_841_600,
                    sha256="1c4958699d4cffd7e777d51421b6044bba300aa0bef286a544309efb082cdbd6",
                ),
                RemoteFile(
                    url=f"{_PDS_LOLA_GDR}/ldem_64.lbl",
                    filename="ldem_64.lbl",
                    size=5_128,
                    sha256="4e7a94c76c716a147b21ccf07669d004246b0d1b017b7d63d50befaa1a34142d",
                ),
            ),
        ),
    ]
}


class DownloadError(RuntimeError):
    """A download failed or produced a file that does not match its record."""


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(part: Path, remote: RemoteFile, log: Callable[[str], None]) -> None:
    size = part.stat().st_size
    if size < remote.size:
        # Keep the partial file so the next run can resume it.
        raise DownloadError(
            f"{remote.filename}: got {size} of {remote.size} bytes; run again to resume"
        )
    if size > remote.size:
        part.unlink()
        raise DownloadError(f"{remote.filename}: got {size} bytes, expected {remote.size}")
    actual = sha256_of(part)
    if remote.sha256 is None:
        log(f"{remote.filename}: sha256 {actual} (not pinned yet)")
    elif actual != remote.sha256:
        part.unlink()
        raise DownloadError(f"{remote.filename}: sha256 {actual}, expected {remote.sha256}")


def download_file(
    remote: RemoteFile,
    dest_dir: Path,
    *,
    opener: Callable = urllib.request.urlopen,
    log: Callable[[str], None] = _log,
) -> Path:
    """Download one file into ``dest_dir`` and return its path.

    A file that already exists with the expected size is left alone.
    """
    target = dest_dir / remote.filename
    if target.exists():
        if target.stat().st_size == remote.size:
            log(f"{remote.filename}: already present, skipping")
            return target
        raise DownloadError(
            f"{target} exists with size {target.stat().st_size}, expected {remote.size}; "
            "delete it to download again"
        )

    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > remote.size:
        part.unlink()
        offset = 0

    if offset < remote.size:
        request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            response = opener(request, timeout=60)
        except URLError as exc:
            raise DownloadError(f"{remote.filename}: {exc}") from exc

        with response:
            if offset and response.status != 206:
                log(f"{remote.filename}: server ignored the resume request, starting over")
                offset = 0
            elif offset:
                log(f"{remote.filename}: resuming at {offset} bytes")

            done = offset
            next_report = 10
            with part.open("ab" if offset else "wb") as fh:
                while chunk := response.read(CHUNK_SIZE):
                    fh.write(chunk)
                    done += len(chunk)
                    percent = 100 * done // remote.size
                    if percent >= next_report:
                        log(f"{remote.filename}: {percent}%")
                        next_report = percent - percent % 10 + 10

    _verify(part, remote, log)
    part.replace(target)
    log(f"{remote.filename}: saved to {target}")
    return target


def download_source(source: Source, dest_dir: Path, **kwargs) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    missing = sum(f.size for f in source.files if not (dest_dir / f.filename).exists())
    free = shutil.disk_usage(dest_dir).free
    if missing > free:
        raise DownloadError(
            f"{source.name} needs {_human(missing)} but only {_human(free)} is free in {dest_dir}"
        )
    return [download_file(f, dest_dir, **kwargs) for f in source.files]


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000:
            return f"{size:.0f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="*", metavar="SOURCE", help="source names to download")
    parser.add_argument("--all", action="store_true", help="download every source")
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="default: data/raw")
    args = parser.parse_args(argv)

    if args.list:
        for s in SOURCES.values():
            print(f"{s.name:<16} {_human(s.size):>8}  {s.description}")
        return 0

    names = list(SOURCES) if args.all else args.sources
    if not names:
        parser.error("name at least one source, or use --all or --list")
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        parser.error(f"unknown source(s): {', '.join(unknown)}; see --list")

    try:
        for name in names:
            download_source(SOURCES[name], args.dest)
    except DownloadError as exc:
        _log(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
