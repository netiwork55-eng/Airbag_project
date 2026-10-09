"""One local capture owner and one immutable recipe snapshot per production run."""
import threading
import time
import uuid
import math
import cv2
import numpy as np
from .schema import State
from .store import atomic_json, now
from .pipeline import Pipeline
from . import ENGINE_VERSION
from .ownership import project_lease


def evidence(store, result, image, maps=None, *, source="test", complete=True):
    key = "event_"+uuid.uuid4().hex
    record = {"id":key,"timestamp":now(),"source":source,"complete":complete,**result}
    if image is not None:
        ok,png = cv2.imencode(".png",image)
        if not ok:
            raise OSError("Evidence image encoding failed")
        record["image_id"] = store.put_image(png.tobytes(),key)["id"]
    if maps:
        np.savez_compressed(store.path("evidence",key,".npz"),**maps)
        record["anomaly_maps"] = list(maps)
    atomic_json(store.path("evidence",key),record)
    return record


class Cycle:
    """Hold the most conservative confirmed result until a stable removal.

    A locator miss is removal evidence; exposure/inference failure is not removal.
    Invalid inspection data is not relabelled as a defect or uncertainty.
    """
    rank = {"PASS":0,"REVIEW":1,"NG":2,"UNABLE_TO_INSPECT":3}

    def __init__(self, stable_frames=3, release_frames=3, release_seconds=.4):
        self.stable_frames,self.release_frames,self.release_seconds = stable_frames,release_frames,release_seconds
        self.current = None
        self.missing_since = None
        self.missing = 0

    def observe(self, result, image, maps, *, absent=False, now_mono=None):
        stamp = time.monotonic() if now_mono is None else now_mono
        if absent:
            if not self.current:
                return None
            if self.missing_since is None:
                self.missing_since = stamp
            self.missing += 1
            if self.missing>=self.release_frames and stamp-self.missing_since>=self.release_seconds:
                return self.finish(True,stamp)
            return None
        self.missing,self.missing_since = 0,None
        if self.current is None:
            self.current = {"frames":0,"result":result,"image":image.copy(),"maps":maps,
                            "committed":False,"pending_state":None,"pending_count":0,"started_mono":stamp}
        self.current["frames"] += 1
        item=self.current
        item["pending_count"] = item["pending_count"]+1 if item["pending_state"]==result["state"] else 1
        item.update(pending_state=result["state"],pending_result=result,pending_image=image.copy(),pending_maps=maps)
        if item["pending_count"]>=self.stable_frames:
            if not item["committed"] or self.rank[result["state"]]>=self.rank[item["result"]["state"]]:
                item.update(result=result,image=image.copy(),maps=maps)
            item["committed"]=True
        return None

    def finish(self, complete, now_mono=None):
        item,self.current = self.current,None
        self.missing,self.missing_since = 0,None
        if item is None:
            return None
        if not complete or item["frames"]<self.stable_frames:
            item["result"] = {**item["result"],"state":State.UNABLE.value,
                              "reasons":["Inspection stopped before a complete stable place-and-remove cycle"]}
        elif not item["committed"] or self.rank[item["pending_state"]]>self.rank[item["result"]["state"]]:
            pending=item["pending_result"]
            state="UNABLE_TO_INSPECT" if pending["state"]=="UNABLE_TO_INSPECT" else "REVIEW"
            item.update(result={**pending,"state":state,"reasons":["No stable inspection decision before the part was removed"]},
                        image=item["pending_image"],maps=item["pending_maps"])
        item["complete"] = complete and item["frames"]>=self.stable_frames
        stamp=time.monotonic() if now_mono is None else now_mono
        item['result']={**item['result'],'cycle_elapsed_ms':round(max(0,stamp-item['started_mono'])*1000,3),
                        'inspected_frames':item['frames'],
                        'cycle_timing_scope':'Observed place-to-removal, including settling and removal confirmation'}
        return item


class OpenCVCamera:
    def __init__(self, config, *, fps=None, buffer_size=None):
        self.config=config
        self.cap = cv2.VideoCapture(config.index,cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"Camera {config.index} cannot be opened; no other camera was selected")
        try:
            if buffer_size is not None:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE,buffer_size)
            if config.width is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,config.width)
            if config.height is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,config.height)
            if fps is not None:
                self.cap.set(cv2.CAP_PROP_FPS,fps)
            # OpenCV exposure units are device-dependent, not microseconds.
            if config.exposure_us is not None or config.gain is not None:
                raise RuntimeError("Microsecond exposure/gain settings require the Basler SDK; configure USB camera controls in its driver")
        except Exception:
            self.cap.release()
            raise

    def read(self):
        ok,frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera did not deliver an image")
        check_dimensions(frame,self.config)
        return frame

    def metadata(self):
        def value(prop):
            v=float(self.cap.get(prop))
            return v if math.isfinite(v) else None
        return {'driver':'opencv','index':self.config.index,
                'width':value(cv2.CAP_PROP_FRAME_WIDTH),'height':value(cv2.CAP_PROP_FRAME_HEIGHT),
                'exposure_driver_units':value(cv2.CAP_PROP_EXPOSURE),'gain':value(cv2.CAP_PROP_GAIN),
                'focus_driver_units':value(cv2.CAP_PROP_FOCUS),'autofocus':value(cv2.CAP_PROP_AUTOFOCUS),
                'note':'Driver-reported values may include unsupported/sentinel values. Focus must be checked on the real image.'}

    def close(self):
        self.cap.release()


class BaslerCamera:
    def __init__(self, config):
        self.config=config
        try:
            from pypylon import pylon
        except ImportError as exc:
            raise RuntimeError("Official Basler pypylon is not installed in this runtime; no custom/fallback driver is used") from exc
        self.pylon = pylon
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        matches = [d for d in devices if d.GetSerialNumber()==config.serial]
        if not config.serial or len(matches)!=1:
            raise RuntimeError("Select the exact Basler camera serial number")
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(matches[0]))
        try:
            self.camera.Open()
            # Do not inherit a cropped sensor ROI from a previous pylon session.
            from pypylon import genicam
            for name in ('OffsetX','OffsetY'):
                node=self.camera.GetNodeMap().GetNode(name)
                if node is not None and genicam.IsWritable(node):
                    node.SetValue(node.GetMin())
                elif node is not None and genicam.IsReadable(node) and node.GetValue()!=0:
                    raise RuntimeError('Camera sensor offset cannot be reset: '+name)
            self.camera.Width.SetValue(config.width if config.width is not None else self.camera.Width.GetMax())
            self.camera.Height.SetValue(config.height if config.height is not None else self.camera.Height.GetMax())
            if config.exposure_us is not None:
                self.camera.ExposureAuto.SetValue("Off")
                self.camera.ExposureTime.SetValue(config.exposure_us)
            if config.gain is not None:
                self.camera.GainAuto.SetValue("Off")
                self.camera.Gain.SetValue(config.gain)
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception:
            self.close()
            raise

    def read(self):
        result = self.camera.RetrieveResult(3000,self.pylon.TimeoutHandling_ThrowException)
        try:
            if not result.GrabSucceeded():
                raise RuntimeError("Basler grab failed: "+result.GetErrorDescription())
            frame=self.converter.Convert(result).GetArray().copy()
            check_dimensions(frame,self.config)
            return frame
        finally:
            result.Release()

    def close(self):
        try:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
        finally:
            if self.camera.IsOpen():
                self.camera.Close()

    def metadata(self):
        from pypylon import genicam
        values={}
        for name in ('Width','Height','OffsetX','OffsetY','BinningHorizontal','BinningVertical',
                     'ExposureTime','ExposureTimeAbs','ExposureAuto','Gain','GainAuto','PixelFormat'):
            node=self.camera.GetNodeMap().GetNode(name)
            if node is not None and genicam.IsReadable(node):
                values[name]=node.GetValue()
        return {'driver':'basler','serial':self.config.serial,'settings':values,
                'focus':'Lens/manual focus must be verified on the real image; not controlled by this application'}


def check_dimensions(frame, config):
    if frame is None or not isinstance(frame,np.ndarray) or frame.ndim!=3 or frame.shape[2]!=3:
        raise RuntimeError('Camera returned an invalid full-frame image')
    if ((config.width is not None and frame.shape[1]!=config.width) or
            (config.height is not None and frame.shape[0]!=config.height)):
        raise RuntimeError(f'Camera delivered {frame.shape[1]}x{frame.shape[0]}, different from requested resolution; no resizing was applied')


class FramePump:
    """Continuously drain capture into one latest-frame slot; never queue old frames."""
    def __init__(self,camera, *, ownership=None):
        self.camera=camera
        self.condition=threading.Condition()
        self.stopped=threading.Event()
        self.latest=None
        self.error=None
        self.sequence=0
        try:
            if ownership is None:
                self.thread=threading.Thread(target=self.read,daemon=True,name="camera-latest-frame")
                self.thread.start()
            else:
                # Capture retains ownership even if its caller times out while stopping.
                self.thread=ownership.start_thread(self.read,daemon=True,name="camera-latest-frame")
        except BaseException:
            self.camera.close()
            raise

    def read(self):
        try:
            while not self.stopped.is_set():
                frame=self.camera.read()
                with self.condition:
                    self.sequence+=1
                    self.latest=(self.sequence,frame,time.monotonic(),now())
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error=str(exc)
                self.latest=None
                self.condition.notify_all()
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                with self.condition:
                    self.error="Camera release failed: "+str(exc)
                    self.condition.notify_all()

    def next(self,previous):
        with self.condition:
            if not self.condition.wait_for(lambda:self.error or (self.latest and self.latest[0]>previous) or self.stopped.is_set(),timeout=4):
                raise RuntimeError("No fresh camera frame received within 4 seconds")
            if self.error:
                raise RuntimeError(self.error)
            if self.stopped.is_set():
                raise RuntimeError("Camera capture stopped")
            return self.latest

    def close(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise RuntimeError("Camera capture did not stop; restart the camera before another run")
        if self.error and self.error.startswith("Camera release failed:"):
            raise RuntimeError(self.error)


class Runtime:
    def __init__(self, store, anomaly, source="production"):
        self.store,self.anomaly = store,anomaly
        self.ownership = project_lease(store.root)
        self.source=source
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.frame = None
        self.capture_pump = None
        self.capture_metadata = None
        self.state = {"running":False,"state":"UNABLE_TO_INSPECT","reasons":["Production is stopped"],"last_event":None}

    @property
    def running(self):
        return ((self.thread is not None and self.thread.is_alive()) or
                (self.capture_pump is not None and self.capture_pump.thread.is_alive()))

    def production_pipeline(self):
        """Read and verify the active recipe without opening a camera or changing Production."""
        active = self.store.production()
        if not active:
            raise ValueError("Validate and activate a recipe before Production")
        if active.get('engine_version')!=ENGINE_VERSION:
            raise ValueError('Inspection software changed; validate the recipe again before Production')
        saved = self.store.validated_recipe(active["recipe_id"],active["report_id"])
        if saved["hash"] != active["recipe_hash"]:
            raise ValueError("Active recipe integrity check failed")
        return self._ready_pipeline(saved)

    def candidate_pipeline(self, recipe_id):
        """Test an explicitly selected immutable recipe without activating it."""
        if self.source != 'candidate':
            raise ValueError('Candidate inspection must be labelled as a test run')
        return self._ready_pipeline(self.store.recipe(recipe_id))

    def _ready_pipeline(self, saved):
        pipeline = Pipeline(self.store,saved,self.anomaly)
        if pipeline.recipe.locator.method == "fixed":
            raise ValueError("Continuous place-and-remove inspection needs a part locator; fixed setup supports image tests only")
        warm,_ = pipeline.run(pipeline.reference)
        if warm["state"]=="UNABLE_TO_INSPECT":
            raise ValueError("Inspection readiness check failed: "+"; ".join(warm["reasons"]))
        return pipeline

    def start(self):
        with self.lock, self.ownership.hold('Web Production inspection'):
            if self.running:
                raise ValueError("Production camera is already running")
            pipeline = self.production_pipeline()
            saved = pipeline.saved
            pump = self.open_capture(pipeline)
            self.stop_event.clear()
            self.frame = None
            self.state = {"running":True,"state":"UNABLE_TO_INSPECT","reasons":["Waiting for the first image"],"recipe_id":saved["id"],"recipe_hash":saved["hash"],"last_event":None}
            try:
                self.thread = self.ownership.start_thread(self.loop,(pump,pipeline),daemon=True,name="production-inspection")
            except BaseException:
                self.state.update(running=False,state='UNABLE_TO_INSPECT',reasons=['Inspection worker could not start'])
                pump.close()
                raise
            return self.snapshot()

    @staticmethod
    def make_cycle(pipeline):
        settings=pipeline.recipe.decision
        return Cycle(settings.stable_frames,settings.release_frames,settings.release_ms/1000)

    def open_capture(self, pipeline, *, terminal_fps=None):
        """Same exact-device acquisition for Web and the existing Terminal window."""
        with self.lock, self.ownership.hold('Live camera setup'):
            if self.running:
                raise ValueError('Camera capture is already running')
            config=pipeline.recipe.camera
            camera=(BaslerCamera(config) if config.driver=='basler' else
                    OpenCVCamera(config,fps=terminal_fps,buffer_size=1 if terminal_fps is not None else None))
            try:
                self.capture_metadata={'requested':config.model_dump(),'actual':camera.metadata(),
                    'timing_scope':'Host read completion; not sensor exposure time'}
            except BaseException:
                camera.close()
                raise
            pump=FramePump(camera,ownership=self.ownership)
            self.capture_pump=pump
            return pump

    def snapshot(self):
        with self.lock:
            return {**self.state,"running":self.running}

    def inspect_frame(self, pipeline, cycle, frame, received, captured_at, sequence):
        """One shared quality/alignment/decision/cycle/evidence path for supplied full frames."""
        result,maps = pipeline.run(frame)
        result.update(captured_at=captured_at,frame_sequence=sequence)
        result['camera_requested']=pipeline.recipe.camera.model_dump()
        if self.capture_metadata is not None:
            result['capture']=self.capture_metadata
        age_ms=(time.monotonic()-received)*1000
        result["frame_age_ms"]=round(age_ms,3)
        if age_ms>pipeline.recipe.camera.max_frame_age_ms:
            result.update(state="UNABLE_TO_INSPECT",reasons=["Inspection image is older than the configured time limit"])
        absent = result["state"]=="UNABLE_TO_INSPECT" and any(s.startswith("Part not found:") for s in result["reasons"])
        closed = cycle.observe(result,frame,maps,absent=absent)
        event = evidence(self.store,closed["result"],closed["image"],closed["maps"],source=self.source,complete=closed["complete"]) if closed else None
        with self.lock:
            held=cycle.current["result"] if cycle.current and cycle.current["committed"] else None
            self.state.update(result,frame_result=result,part_present=not absent,cycle_result=held,
                              cycle_state=held["state"] if held else None)
            if result["state"]=="UNABLE_TO_INSPECT":
                self.state.update(state=result["state"],reasons=result["reasons"])
            elif held and not absent:
                self.state.update(state=held["state"],reasons=held["reasons"]+["Result held for this part; lift the part to complete"])
                if self.rank_pending(result["state"],held["state"]):
                    self.state.update(state="REVIEW",reasons=["Waiting to confirm a change in this part's inspection result"])
            elif not absent:
                self.state.update(state="UNABLE_TO_INSPECT",reasons=["Waiting for a stable inspection result"])
            if event:
                self.state["last_event"] = event["id"]
        return self.state.copy(), result, maps

    def finish_cycle(self, cycle):
        """An interrupted part can never be recorded as a complete PASS."""
        closed = cycle.finish(False)
        if closed:
            event = evidence(self.store,closed["result"],closed["image"],closed["maps"],
                             source=self.source,complete=False)
            self.state["last_event"] = event["id"]

    def loop(self, pump, pipeline):
        cycle = self.make_cycle(pipeline)
        sequence = 0
        try:
            while not self.stop_event.is_set():
                sequence,frame,received,captured_at = pump.next(sequence)
                self.inspect_frame(pipeline,cycle,frame,received,captured_at,sequence)
                ok,jpg = cv2.imencode(".jpg",frame)
                if not ok:
                    raise RuntimeError("Camera image encoding failed")
                with self.lock:
                    self.frame = jpg.tobytes()
                self.stop_event.wait(.03)
        except Exception as exc:
            with self.lock:
                self.frame = None
                self.state.update(state="UNABLE_TO_INSPECT",reasons=[str(exc)])
        finally:
            try:
                self.finish_cycle(cycle)
            except Exception as exc:
                self.state.update(state="UNABLE_TO_INSPECT",reasons=["Evidence could not be stored: "+str(exc)])
            finally:
                try:
                    pump.close()
                except Exception as exc:
                    self.state.update(state="UNABLE_TO_INSPECT",reasons=[str(exc)])
                with self.lock:
                    self.state["running"] = False
                    self.frame = None

    @staticmethod
    def rank_pending(current,held):
        return Cycle.rank[current]>Cycle.rank[held]

    def stop(self):
        self.stop_event.set()
        thread = self.thread
        if thread:
            thread.join(timeout=6)
            if thread.is_alive():
                self.anomaly.close()
                thread.join(timeout=6)
            if thread.is_alive():
                raise RuntimeError("Camera worker is still stopping; camera remains locked")
        if self.running:
            raise RuntimeError("Camera capture is still stopping; the camera remains locked")
        with self.lock:
            self.state.update(running=False,state="UNABLE_TO_INSPECT",reasons=["Production is stopped"])
            self.frame = None
