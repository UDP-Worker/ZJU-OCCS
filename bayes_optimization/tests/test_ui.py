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
    vols = res["voltages"]
    vols[2] *= 0.5
    vols[44] *= 0.5
    resp = client.post("/manual", json={"voltages": vols})
    data = resp.json()
    assert len(data["response"]) == len(data["ideal"])
