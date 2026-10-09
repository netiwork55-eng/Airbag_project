"""Bounded local subprocess protocol. Framework startup is outside web/camera imports."""
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import zlib
import cv2
import numpy as np


class AnomalyClient:
    def __init__(self, root, store):
        self.root,self.store = Path(root),store
        self.lock = threading.RLock()
        self.process = None
        self.output = queue.Queue()
        self.log = None
        self.cancelled = threading.Event()

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        python = self.root / ".venv-inspection" / "Scripts" / "python.exe"
        if not python.is_file():
            raise RuntimeError("Anomalib worker is not installed; follow the local setup guide")
        self.output = queue.Queue()
        self.log = (self.store.root / "worker.log").open("a",encoding="utf-8")
        self.process = subprocess.Popen([str(python),"-u","-m","inspection.anomaly_worker",str(self.store.root)],
            cwd=self.root/"07_web_app",stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,encoding="utf-8",creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        proc,out = self.process,self.output
        def read():
            try:
                for line in proc.stdout:
                    out.put(line)
            except (ValueError,OSError):
                pass  # Cancellation can close stdout while a read is pending.
            finally:
                out.put(None)
        threading.Thread(target=read,daemon=True,name="anomalib-output").start()

    def call(self, request, timeout=90):
        with self.lock:
            if self.cancelled.is_set():
                raise RuntimeError("Worker request was cancelled")
            self.start()
            if self.cancelled.is_set():
                self.close()
                raise RuntimeError("Worker request was cancelled")
            try:
                self.process.stdin.write(json.dumps(request)+"\n")
                self.process.stdin.flush()
                line = self.output.get(timeout=timeout)
                if line is None:
                    raise RuntimeError("Anomalib worker exited; see the local worker log")
                response = json.loads(line)
                if not response["ok"]:
                    raise RuntimeError(response["error"])
                return response["result"]
            except (queue.Empty,BrokenPipeError,json.JSONDecodeError) as exc:
                self.close()
                raise RuntimeError("Anomalib inference timed out or the worker failed") from exc

    def cancel(self):
        self.cancelled.set()
        self.close()

    def predict(self, model_id, image):
        ok,encoded = cv2.imencode(".png",image)
        if not ok:
            raise ValueError("Could not encode inspection area")
        response = self.call({"action":"predict","model_id":model_id,"image":base64.b64encode(encoded).decode()})
        if 'map' in response:
            response["map"] = np.frombuffer(zlib.decompress(base64.b64decode(response["map"])),np.float32).reshape(response["shape"])
        return response

    def close(self):
        process,self.process = self.process,None
        if process is not None:
            if process.poll() is None:
                if os.name == "nt":
                    # A Windows venv launcher can own a second Python process.
                    # Terminating only the launcher leaves inference/training orphaned.
                    taskkill = Path(os.environ["SystemRoot"]) / "System32" / "taskkill.exe"
                    try:
                        subprocess.run([str(taskkill),"/PID",str(process.pid),"/T","/F"],
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,
                            creationflags=subprocess.CREATE_NO_WINDOW,check=False)
                    except (OSError,subprocess.TimeoutExpired):
                        process.kill()
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin,process.stdout):
                if stream:
                    try:
                        stream.close()
                    except (OSError,ValueError):
                        pass
        if self.log:
            self.log.close()
            self.log = None
