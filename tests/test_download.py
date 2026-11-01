import hashlib
import re
import threading
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lunar_crater.data import download
from lunar_crater.data.download import DownloadError, RemoteFile, Source

PAYLOAD = bytes(range(256)) * 40_000  # ~10 MB, several chunks
PAYLOAD_SIZE = len(PAYLOAD)
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    payload = PAYLOAD
    honour_range = True
    ranges: list[str] = []

    def do_GET(self):
        body, status = self.payload, 200
        requested = self.headers.get("Range")
        type(self).ranges.append(requested)
        if requested and self.honour_range:
            start = int(re.fullmatch(r"bytes=(\d+)-", requested).group(1))
            body, status = body[start:], 206
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


Server = namedtuple("Server", "url handler")


@pytest.fixture
def server():
    handler = type("Handler", (_Handler,), {"ranges": []})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield Server(f"http://127.0.0.1:{httpd.server_address[1]}/file.bin", handler)
    httpd.shutdown()
    httpd.server_close()


def remote(url, *, size=PAYLOAD_SIZE, sha256=PAYLOAD_SHA):
    return RemoteFile(url=url, filename="file.bin", size=size, sha256=sha256)


def quiet(_message):
    pass


def test_downloads_and_verifies(server, tmp_path):
    path = download.download_file(remote(server.url), tmp_path, log=quiet)

    assert path.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.bin.part").exists()
    assert server.handler.ranges == [None]


def test_logs_checksum_when_not_pinned(server, tmp_path):
    messages = []
    download.download_file(remote(server.url, sha256=None), tmp_path, log=messages.append)

    assert any(PAYLOAD_SHA in m for m in messages)


def test_resumes_partial_download(server, tmp_path):
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:3_000_000])

    path = download.download_file(remote(server.url), tmp_path, log=quiet)

    assert path.read_bytes() == PAYLOAD
    assert server.handler.ranges == ["bytes=3000000-"]


def test_restarts_when_server_ignores_range(server, tmp_path):
    server.handler.honour_range = False
    (tmp_path / "file.bin.part").write_bytes(b"stale bytes")

    path = download.download_file(remote(server.url), tmp_path, log=quiet)

    assert path.read_bytes() == PAYLOAD


def test_checksum_mismatch_discards_file(server, tmp_path):
    with pytest.raises(DownloadError, match="sha256"):
        download.download_file(remote(server.url, sha256="0" * 64), tmp_path, log=quiet)

    assert list(tmp_path.iterdir()) == []


def test_oversized_download_discards_file(server, tmp_path):
    with pytest.raises(DownloadError, match="expected 100"):
        download.download_file(remote(server.url, size=100), tmp_path, log=quiet)

    assert list(tmp_path.iterdir()) == []


def test_skips_existing_file_of_expected_size(tmp_path):
    (tmp_path / "file.bin").write_bytes(PAYLOAD)

    def no_network(*args, **kwargs):
        raise AssertionError("should not download")

    path = download.download_file(remote("http://unused"), tmp_path, opener=no_network, log=quiet)

    assert path == tmp_path / "file.bin"


def test_existing_file_of_wrong_size_is_an_error(tmp_path):
    (tmp_path / "file.bin").write_bytes(b"short")

    with pytest.raises(DownloadError, match="delete it"):
        download.download_file(remote("http://unused"), tmp_path, log=quiet)


def test_refuses_when_disk_is_too_small(monkeypatch, tmp_path):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _: usage(0, 0, 10))
    source = Source("big", "too big", (remote("http://unused"),))

    with pytest.raises(DownloadError, match="needs"):
        download.download_source(source, tmp_path, log=quiet)


def test_registry_records_are_well_formed():
    files = [f for s in download.SOURCES.values() for f in s.files]

    assert len({f.filename for f in files}) == len(files)
    for f in files:
        assert f.url.startswith("https://")
        assert f.size > 0
        assert f.sha256 is None or re.fullmatch(r"[0-9a-f]{64}", f.sha256)


def test_cli_lists_sources(capsys):
    assert download.main(["--list"]) == 0

    out = capsys.readouterr().out
    assert all(name in out for name in download.SOURCES)


@pytest.mark.parametrize("argv", [[], ["no_such_source"]])
def test_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        download.main(argv)
