"""One OS-backed project lease shared by existing camera/workflow entry points."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import threading
from filelock import FileLock, Timeout
from .store import atomic_json, now


class InspectionBusy(ValueError):
    pass


class ProjectLease:
    def __init__(self, data_root):
        self.root = Path(data_root)
        self.path = self.root / '.runtime.lock'
        self.owner_path = self.root / '.runtime-owner.json'
        # Work starts on a request thread and finishes on the existing worker thread.
        self.lock = FileLock(str(self.path), timeout=0, thread_local=False)
        self.guard = threading.RLock()

    def acquire(self, purpose):
        with self.guard:
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                self.lock.acquire()
            except Timeout as exc:
                raise InspectionBusy('Another process owns inspection/camera work; stop that run before continuing') from exc
            try:
                if self.lock.lock_counter == 1:
                    atomic_json(self.owner_path, {'pid':os.getpid(),'purpose':purpose,'started_at':now()})
            except Exception:
                self.lock.release()
                raise

    def release(self):
        with self.guard:
            if self.lock.lock_counter == 1:
                try:
                    self.owner_path.unlink(missing_ok=True)
                finally:
                    self.lock.release()
            else:
                self.lock.release()

    @contextmanager
    def hold(self, purpose):
        self.acquire(purpose)
        try:
            yield
        finally:
            self.release()

    def external_owner(self):
        # Stale metadata after a crash is not ownership: the OS lock is authoritative.
        with self.guard:
            if self.lock.is_locked or not self.path.exists():
                return None
            probe = FileLock(str(self.path), timeout=0)
            try:
                with probe:
                    return None
            except Timeout:
                try:
                    return json.loads(self.owner_path.read_text(encoding='utf-8'))
                except (OSError,ValueError):
                    return {'purpose':'Inspection work in another process','pid':None}

    def start_thread(self, target, args=(), **kwargs):
        self.acquire('Web inspection/training worker')
        def run():
            try:
                target(*args)
            finally:
                self.release()
        try:
            thread = threading.Thread(target=run, **kwargs)
            thread.start()
        except BaseException:
            self.release()
            raise
        return thread


_leases = {}
_guard = threading.Lock()


def project_lease(data_root):
    key = str(Path(data_root).resolve()).casefold()
    with _guard:
        if key not in _leases:
            _leases[key] = ProjectLease(data_root)
        return _leases[key]
