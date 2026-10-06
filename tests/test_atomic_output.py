"""Fault injection for the private, single-file output commit boundary."""

from __future__ import annotations

import errno
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from convmerge import _output


class StreamProxy:
    def __init__(self, stream, fault):
        self.stream = stream
        self.fault = fault

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, text):
        self.stream.write(text)
        if self.fault == "write":
            raise OSError(errno.ENOSPC, "injected write failure")
        return len(text)

    def flush(self):
        if self.fault == "flush":
            raise OSError("injected flush failure")
        return self.stream.flush()

    def close(self):
        self.stream.close()
        if self.fault == "close":
            raise OSError("injected close failure")


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "fault", ["body", "interrupt", "write", "flush", "fsync", "close", "replace", "fdopen", "chmod"]
)
def test_failures_never_publish_partial_file(tmp_path, monkeypatch, existing, fault):
    target = tmp_path / "out.jsonl"
    if existing:
        target.write_bytes(b"old\n")
    original_fdopen = os.fdopen

    def fdopen(*args, **kwargs):
        if fault == "fdopen":
            raise OSError("injected fdopen failure")
        return StreamProxy(original_fdopen(*args, **kwargs), fault)

    def fail(*args, **kwargs):
        raise OSError(f"injected {fault} failure")

    monkeypatch.setattr(os, "fdopen", fdopen)
    if fault in {"fsync", "replace", "chmod"}:
        monkeypatch.setattr(os, fault, fail)
    # chmod only runs when preserving existing permissions.
    will_fail = existing or fault != "chmod"
    expected = KeyboardInterrupt if fault == "interrupt" else OSError
    if will_fail:
        with pytest.raises(expected):
            with _output.atomic_text_writer(target) as stream:
                stream.write("new\n")
                assert target.read_bytes() == b"old\n" if existing else not target.exists()
                if fault == "body":
                    raise OSError("injected processing failure")
                if fault == "interrupt":
                    raise KeyboardInterrupt()
        assert target.read_bytes() == b"old\n" if existing else not target.exists()
    else:
        with _output.atomic_text_writer(target) as stream:
            stream.write("new\n")
        assert target.read_text() == "new\n"
    assert not list(tmp_path.glob(".convmerge-*.tmp"))


def test_flush_sync_close_precede_replace(tmp_path, monkeypatch):
    target = tmp_path / "out"
    order = []
    real_open, real_sync, real_replace = os.fdopen, os.fsync, os.replace

    class Traced(StreamProxy):
        def flush(self):
            order.append("flush")
            return self.stream.flush()

        def close(self):
            order.append("close")
            return self.stream.close()

    def sync(fd):
        order.append("fsync")
        return real_sync(fd)

    def replace(src, dst):
        order.append("replace")
        assert not target.exists()
        assert Path(src).parent == target.parent.resolve()
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fdopen", lambda *a, **k: Traced(real_open(*a, **k), None))
    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(os, "replace", replace)
    with _output.atomic_text_writer(target) as stream:
        stream.write("done")
    assert order == ["flush", "fsync", "close", "replace"]
    assert target.read_text() == "done"


def test_cleanup_error_does_not_hide_original(tmp_path, monkeypatch, caplog):
    target = tmp_path / "out"
    target.write_bytes(b"old")
    original_unlink = Path.unlink

    def refuse_temporary(path, *args, **kwargs):
        if path.name.startswith(".convmerge-"):
            raise PermissionError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", refuse_temporary)
        with pytest.raises(RuntimeError, match="primary failure"):
            with _output.atomic_text_writer(target) as stream:
                stream.write("partial")
                raise RuntimeError("primary failure")
    assert target.read_bytes() == b"old"
    leftovers = list(tmp_path.glob(".convmerge-*.tmp"))
    assert len(leftovers) == 1
    assert str(leftovers[0]) in caplog.text
    leftovers[0].unlink()


def test_failed_fdopen_closes_descriptor(tmp_path, monkeypatch):
    observed = []
    original = _output._new_file

    def tracked(target):
        path, fd = original(target)
        observed.append(fd)
        return path, fd

    def fail(*args, **kwargs):
        raise OSError("open failed")

    monkeypatch.setattr(_output, "_new_file", tracked)
    monkeypatch.setattr(os, "fdopen", fail)
    with pytest.raises(OSError, match="open failed"):
        with _output.atomic_text_writer(tmp_path / "out"):
            pass
    with pytest.raises(OSError):
        os.fstat(observed[0])
    assert not list(tmp_path.iterdir())


def test_output_symlink_keeps_link_and_replaces_target(tmp_path):
    target, link = tmp_path / "real", tmp_path / "alias"
    target.write_bytes(b"old")
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    with _output.atomic_text_writer(link) as stream:
        stream.write("new")
        assert target.read_bytes() == b"old"
    assert link.is_symlink()
    assert target.read_bytes() == b"new"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_permissions_and_new_file_umask(tmp_path):
    target = tmp_path / "out"
    target.write_bytes(b"old")
    target.chmod(0o640)
    with _output.atomic_text_writer(target) as stream:
        stream.write("new")
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    # Change umask only in an isolated process, never in the host process.
    script = (
        "import os,sys,stat; from convmerge._output import atomic_text_writer; "
        "from pathlib import Path; os.umask(0o027); p=Path(sys.argv[1]); "
        "\nwith atomic_text_writer(p) as f: f.write('new')\n"
        "assert stat.S_IMODE(p.stat().st_mode)==0o640"
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path / "new")], check=True)


def test_readonly_and_directory_outputs_refused(tmp_path):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(OSError):
        with _output.atomic_text_writer(directory):
            pass
    readonly = tmp_path / "readonly"
    readonly.write_bytes(b"old")
    readonly.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            with _output.atomic_text_writer(readonly):
                pass
        assert readonly.read_bytes() == b"old"
    finally:
        readonly.chmod(0o600)


@pytest.mark.skipif(os.name == "nt", reason="POSIX FIFO")
def test_fifo_is_not_replaced(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(OSError):
        with _output.atomic_text_writer(fifo):
            pass
    assert stat.S_ISFIFO(fifo.stat().st_mode)


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL before commit")
@pytest.mark.parametrize("existing", [False, True])
def test_killed_writer_never_publishes_partial_output(tmp_path, existing):
    target, ready = tmp_path / "out", tmp_path / "ready"
    if existing:
        target.write_bytes(b"old")
    script = (
        "import sys,time; from pathlib import Path; "
        "from convmerge._output import atomic_text_writer\n"
        "with atomic_text_writer(sys.argv[1]) as f:\n"
        " f.write('partial'); f.flush(); Path(sys.argv[2]).write_text('ready'); time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script, str(target), str(ready)])
    try:
        end = time.monotonic() + 10
        while not ready.exists() and proc.poll() is None and time.monotonic() < end:
            time.sleep(0.01)
        assert ready.exists(), "child never reached the pre-commit checkpoint"
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        assert target.read_bytes() == b"old" if existing else not target.exists()
        # SIGKILL cannot run cleanup: isolated remnants are expected, not final outputs.
        assert list(tmp_path.glob(".convmerge-*.tmp"))
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)


def test_unowned_temp_collision_is_never_deleted(tmp_path, monkeypatch):
    target = tmp_path / "out"
    target.write_bytes(b"old")
    collision = tmp_path / ".convmerge-fixed.tmp"
    collision.write_bytes(b"belongs to someone else")
    monkeypatch.setattr(_output.secrets, "token_hex", lambda n: "fixed")
    with pytest.raises(FileExistsError):
        with _output.atomic_text_writer(target):
            pass
    assert target.read_bytes() == b"old"
    assert collision.read_bytes() == b"belongs to someone else"


@pytest.mark.parametrize("text", ["a\nb\n", "a\r\nb\r\n", "한글\n", "\ufeffhead\nend"])
def test_text_writer_matches_native_open_bytes(tmp_path, text):
    control, target = tmp_path / "control", tmp_path / "out"
    control.write_text(text, encoding="utf-8")
    with _output.atomic_text_writer(target) as stream:
        stream.write(text)
    assert target.read_bytes() == control.read_bytes()
