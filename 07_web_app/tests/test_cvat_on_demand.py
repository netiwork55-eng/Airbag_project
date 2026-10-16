"""Application startup must not start optional annotation polling."""
import asyncio
import threading
import sys
from pathlib import Path
from unittest.mock import Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_startup_does_not_start_cvat_thread_or_contact_connector(monkeypatch):
    import app
    thread=Mock(side_effect=AssertionError('Unexpected background thread'))
    pull=Mock(side_effect=AssertionError('Unexpected CVAT request'))
    monkeypatch.setattr(app.threading,'Thread',thread)
    monkeypatch.setattr(app,'pull_cvat_annotations',pull)
    monkeypatch.setattr(app,'sync_confirmed_reviews_to_cvat',pull)
    monkeypatch.setattr(app.vision_service,'close',lambda:None)
    monkeypatch.setattr(app,'TRAIN_PROCESS',None)
    monkeypatch.setattr(app,'REFERENCE_TRAIN_STOP_EVENT',threading.Event())
    async def lifecycle():
        async with app.lifespan(app.app):
            pass
    asyncio.run(lifecycle())
    thread.assert_not_called()
    pull.assert_not_called()
