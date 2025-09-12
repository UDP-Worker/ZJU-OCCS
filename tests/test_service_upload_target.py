import io
import numpy as np
import pytest


def _make_app():
    fastapi = pytest.importorskip("fastapi")
    from OCCS.service.app import create_app
    return create_app()


def _make_two_row_csv_bytes(n=32):
    lam = np.linspace(1.55e-6, 1.56e-6, n)
    t = np.sin(np.linspace(0, 2*np.pi, n))
    buf = io.StringIO()
    buf.write(",".join(f"{x:.9e}" for x in lam) + "\n")
    buf.write(",".join(f"{y:.6f}" for y in t) + "\n")
    return buf.getvalue().encode('utf-8')


def test_upload_and_use_target_csv():
    from fastapi.testclient import TestClient

    app = _make_app()
    client = TestClient(app)

    data = _make_two_row_csv_bytes(48)
    r = client.post('/api/upload/target', files={'file': ('ideal.csv', data, 'text/csv')})
    assert r.status_code == 200
    path = r.json()['path']
    assert path.endswith('.csv')

    payload = {
        'backend': 'mock',
        'dac_size': 2,
        'wavelength': {'start': 1.55e-6, 'stop': 1.56e-6, 'M': 48},
        'target_csv_path': path,
    }
    sid = client.post('/api/session', json=payload).json()['session_id']

    wf = client.get(f'/api/session/{sid}/response').json()
    assert len(wf['lambda']) == 48 and len(wf['target']) == 48

