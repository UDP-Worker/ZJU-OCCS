import numpy as np
from fastapi.testclient import TestClient
from bayes_optimization.ui.server import app
from bayes_optimization.bayes_optimizer import config


def test_visual_workflow():
    client = TestClient(app)
    # 设置电极数量
    resp = client.post("/set_channels", json={"num_channels": 64})
    assert resp.json()["num_channels"] == 64

    # 上传自定义方波波形
    wl = np.linspace(1.5e-6, 1.6e-6, 50)
    wave = np.where(np.arange(50) < 25, -30.0, -20.0)
    text = ",".join(map(str, wl)) + "\n" + ",".join(map(str, wave))
    resp = client.post("/upload_waveform", files={"file": ("wave.csv", text)})
    assert resp.status_code == 200

    # 标定
    resp = client.post("/calibrate")
    assert resp.json()["modes"] >= 1

    # 运行优化，减少步数加快测试
    config.BO_MAX_STEPS = 3
    config.SPSA_STEPS = 2
    res = client.post("/optimize").json()
    assert len(res["voltages"]) == 64

    # 手动调整部分电极
    vols = res["voltages"].copy()
    vols[2] *= 0.5
    vols[44] *= 0.5
    resp = client.post("/manual", json={"voltages": vols})
    data = resp.json()
    assert len(data["response"]) == len(data["ideal"])
    status = client.get("/status").json()
    # optimized voltages should persist after manual adjustment
    assert np.allclose(status["voltages"], res["voltages"])
    assert not np.allclose(status["voltages"], vols)

    # 再次运行优化和手调，确保接口仍能更新
    res2 = client.post("/optimize").json()
    assert len(res2["voltages"]) == 64
    vols2 = res2["voltages"].copy()
    vols2[0] += 0.1
    resp = client.post("/manual", json={"voltages": vols2})
    assert resp.status_code == 200


def test_real_mode_requires_connection():
    client = TestClient(app)
    r = client.post("/set_mode", json={"mode": "real"})
    assert r.json()["mode"] == "real"
    assert not r.json()["connected"]
    assert client.post("/calibrate").status_code == 400
    assert client.post("/optimize").status_code == 400
    assert client.post("/manual", json={"voltages": [0.0]}).status_code == 400
