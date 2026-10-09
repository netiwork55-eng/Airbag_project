"""Never contend with a user's live camera while exercising legacy web guards."""
import sys
import pytest


@pytest.fixture(autouse=True)
def isolate_legacy_runtime_ownership(tmp_path,monkeypatch):
    app=sys.modules.get('app')
    if app is not None and hasattr(app,'vision_service'):
        from inspection.ownership import project_lease
        monkeypatch.setattr(app.vision_service.runtime,'ownership',project_lease(tmp_path/'ownership'))
