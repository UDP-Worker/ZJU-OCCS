from pathlib import Path


def test_webui_scaffold_files_exist():
    base = Path(__file__).resolve().parents[1] / 'OCCS' / 'webui'
    assert base.exists(), 'OCCS/webui should exist'
    # key files
    assert (base / 'index.html').exists()
    assert (base / 'vite.config.ts').exists()
    assert (base / 'src' / 'main.tsx').exists()
    assert (base / 'src' / 'App.tsx').exists()
    # api and types
    assert (base / 'src' / 'api' / 'client.ts').exists()
    assert (base / 'src' / 'api' / 'ws.ts').exists()
    assert (base / 'src' / 'types.ts').exists()
    # components
    for name in [
        'BackendSelector',
        'SessionForm',
        'VoltagePanel',
        'WaveformChart',
        'LossChart',
        'OptimizerControls',
        'StatusBar',
    ]:
        assert (base / 'src' / 'components' / f'{name}.tsx').exists()


def test_ws_hook_has_expected_path():
    base = Path(__file__).resolve().parents[1] / 'OCCS' / 'webui' / 'src' / 'api'
    text = (base / 'ws.ts').read_text(encoding='utf-8')
    assert '/api/session/' in text and '/stream' in text


def test_static_mount_serves_index():
    import pytest
    fastapi = pytest.importorskip('fastapi')
    from fastapi.testclient import TestClient
    from OCCS.service.app import create_app

    app = create_app()
    client = TestClient(app)
    # Only checks that static mount works when dist exists
    r = client.get('/')
    assert r.status_code == 200
    assert 'OCCS Web UI' in r.text

