"""Cooperative V2/V3 microphone handoff; no audio or filesystem I/O on import.

Protocol v1: per-process registration is held for this run, and the shared
microphone lease is held until the stream actually closes. PID text is not proof;
peers must observe the OS lock on the registration belonging to each live process.
"""
import fcntl
import os
from pathlib import Path


class Lease:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def acquire(self):
        if self.handle is not None:
            return False
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        handle = os.fdopen(fd, 'a+')
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        except BaseException:
            handle.close()
            raise
        self.handle = handle
        return True

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class LeasedStream:
    def __init__(self, stream, lease):
        self.stream, self.lease = stream, lease
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def close(self):
        if self.closed:
            return
        self.stream.close()  # Failure must retain the microphone lease.
        self.lease.close()
        self.closed = True


class SupervisorHandoff:
    def __init__(self, controller, controller_running, *, pid=None):
        path = Path(controller).expanduser().absolute()
        self.controller_running = controller_running
        self.registration = Lease(path.with_name(path.name + '.v3-supervisor-v1-' + str(pid or os.getpid())))
        self.microphone = Lease(path.with_name(path.name + '.v3-microphone'))

    def start(self):
        if not self.registration.acquire():
            raise RuntimeError('SupervisorHandoffAlreadyRegistered')

    def open_stream(self, factory):
        if self.controller_running() or not self.microphone.acquire():
            return None
        try:
            if self.controller_running():
                self.microphone.close()
                return None
            return LeasedStream(factory(), self.microphone)
        except BaseException:
            self.microphone.close()  # Factory must not return a partially opened stream.
            raise

    def close(self):
        if self.microphone.handle is not None:
            raise RuntimeError('SupervisorMicrophoneNotClosed')
        self.registration.close()
