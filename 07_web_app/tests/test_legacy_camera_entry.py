"""Existing utility delegates to the one camera entry; never open hardware here."""
from pathlib import Path
import sys
from unittest.mock import Mock

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'03_dataset'))
sys.path.insert(0,str(ROOT/'01_camera'))
import airbag_negative_mining_pipeline as mining
import airbag_webcam_detection as camera


def test_negative_mining_camera_forwards_only_camera_arguments(monkeypatch):
    entry=Mock()
    monkeypatch.setattr(camera,'main',entry)
    monkeypatch.setattr(sys,'argv',['negative-mining','--camera'])
    mining.main()
    entry.assert_called_once_with(['--camera-index','0'])
    args=camera.parse_args(entry.call_args.args[0])
    assert args.camera_index==0 and args.candidate_recipe is None


def test_negative_mining_preserves_explicit_camera_selection(monkeypatch):
    entry=Mock()
    monkeypatch.setattr(camera,'main',entry)
    mining.run_camera_inference(2)
    entry.assert_called_once_with(['--camera-index','2'])
