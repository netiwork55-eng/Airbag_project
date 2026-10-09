"""Camera adapter contract tests; mocks do not certify a physical camera."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inspection import runtime
from inspection.schema import Camera


def test_usb_resolution_mismatch_is_error_not_resized(monkeypatch):
    cap=Mock()
    cap.isOpened.return_value=True
    cap.read.return_value=(True,np.zeros((480,640,3),np.uint8))
    monkeypatch.setattr(runtime.cv2,'VideoCapture',lambda *args:cap)
    camera=runtime.OpenCVCamera(Camera(width=1920,height=1080))
    try:
        with pytest.raises(RuntimeError,match='different from requested'):
            camera.read()
    finally:camera.close()
    cap.release.assert_called_once()


def test_usb_setup_exception_releases_camera(monkeypatch):
    cap=Mock()
    cap.isOpened.return_value=True
    cap.set.side_effect=RuntimeError('Driver failed')
    monkeypatch.setattr(runtime.cv2,'VideoCapture',lambda *args:cap)
    with pytest.raises(RuntimeError,match='Driver failed'):
        runtime.OpenCVCamera(Camera(width=1920))
    cap.release.assert_called_once()


def test_basler_copy_survives_released_result_and_error_also_releases():
    camera=runtime.BaslerCamera.__new__(runtime.BaslerCamera)
    camera.config=Camera(driver='basler',serial='mock')
    camera.pylon=SimpleNamespace(TimeoutHandling_ThrowException=1)
    raw=np.full((20,30,3),123,np.uint8)
    grab=Mock()
    grab.GrabSucceeded.return_value=True
    grab.Release.side_effect=lambda:raw.fill(0)
    camera.camera=Mock()
    camera.camera.RetrieveResult.return_value=grab
    camera.converter=Mock()
    camera.converter.Convert.return_value.GetArray.return_value=raw
    frame=camera.read()
    assert frame.min()==123 and raw.max()==0
    grab.Release.assert_called_once()
    grab.reset_mock()
    grab.GrabSucceeded.return_value=False
    grab.GetErrorDescription.return_value='disconnect'
    with pytest.raises(RuntimeError,match='disconnect'):camera.read()
    grab.Release.assert_called_once()


def test_basler_close_runs_even_when_stop_raises():
    camera=runtime.BaslerCamera.__new__(runtime.BaslerCamera)
    camera.camera=Mock()
    camera.camera.IsGrabbing.return_value=True
    camera.camera.IsOpen.return_value=True
    camera.camera.StopGrabbing.side_effect=RuntimeError('stop failed')
    with pytest.raises(RuntimeError,match='stop failed'):camera.close()
    camera.camera.Close.assert_called_once()


def test_cycle_timing_counts_place_to_removal():
    cycle=runtime.Cycle(stable_frames=2,release_frames=2,release_seconds=.1)
    frame=np.zeros((16,16,3),np.uint8)
    cycle.observe({'state':'PASS'},frame,{},now_mono=10)
    cycle.observe({'state':'PASS'},frame,{},now_mono=10.1)
    cycle.observe({},frame,{},absent=True,now_mono=11)
    closed=cycle.observe({},frame,{},absent=True,now_mono=11.2)
    assert closed['result']['cycle_elapsed_ms']==1200
    assert closed['result']['inspected_frames']==2
