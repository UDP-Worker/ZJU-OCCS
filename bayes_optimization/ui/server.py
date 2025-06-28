from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayes_optimization.bayes_optimizer import (
    config,
    models,
    acquisition,
    optimizer,
    spsa,
)
from bayes_optimization.bayes_optimizer import calibrator
from bayes_optimization.bayes_optimizer.simulate import optical_chip

app = FastAPI()

static_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=static_dir), name="static")

CURRENT_MODE = "mock"
CALIBRATION = None
CURRENT_VOLTAGES = np.zeros(config.NUM_CHANNELS)


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/status")
def get_status():
    return {
        "mode": CURRENT_MODE,
        "num_channels": config.NUM_CHANNELS,
        "voltages": CURRENT_VOLTAGES.tolist(),
    }


@app.post("/set_mode")
def set_mode(data: dict):
    global CURRENT_MODE
    mode = data.get("mode")
    if mode not in {"mock", "real"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    CURRENT_MODE = mode
    return {"mode": CURRENT_MODE}


@app.post("/set_channels")
def set_channels(data: dict):
    num = int(data.get("num_channels", config.NUM_CHANNELS))
    config.NUM_CHANNELS = num
    from bayes_optimization.bayes_optimizer.hardware.mock_hardware import MockOSA
    MockOSA.current_volts = np.zeros(num)
    global CURRENT_VOLTAGES
    CURRENT_VOLTAGES = np.zeros(num)
    return {"num_channels": num}


@app.post("/upload_waveform")
async def upload_waveform(file: UploadFile = File(...)):
    text = (await file.read()).decode()
    lines = text.strip().splitlines()
    if len(lines) < 2:
        raise HTTPException(status_code=400, detail="bad file")
    wl = np.fromstring(lines[0], sep=",", dtype=float)
    resp = np.fromstring(lines[1], sep=",", dtype=float)
    optical_chip.set_target_waveform(wl, resp)
    return {"points": len(wl)}


@app.post("/calibrate")
def run_calibrate():
    global CALIBRATION
    J = calibrator.measure_jacobian()
    n, mat = calibrator.compress_modes(J)
    CALIBRATION = {"modes": n, "matrix": mat.tolist()}
    return {"modes": n}


def loss_fn(volts: np.ndarray) -> float:
    _, resp = optical_chip.response(volts)
    return float(np.mean((resp - optical_chip._IDEAL_RESPONSE) ** 2))


@app.post("/manual")
def manual_adjust(data: dict):
    volts = np.array(data.get("voltages", []), dtype=float)
    w, resp = optical_chip.response(volts)
    global CURRENT_VOLTAGES
    CURRENT_VOLTAGES = volts.copy()
    return {
        "wavelengths": w.tolist(),
        "response": resp.tolist(),
        "ideal": optical_chip._IDEAL_RESPONSE.tolist(),
        "voltages": CURRENT_VOLTAGES.tolist(),
    }


@app.post("/optimize")
def run_optimize():
    num_ch = config.NUM_CHANNELS
    bounds = np.tile(config.V_RANGE, (num_ch, 1))
    start = np.full(num_ch, sum(config.V_RANGE) / 2)

    gp = models.GaussianProcess()
    bo = optimizer.BayesOptimizer(gp, acquisition.expected_improvement, bounds)

    bo_res = bo.optimize(start, loss_fn, steps=config.BO_MAX_STEPS)
    refined = spsa.spsa_refine(bo_res["best_x"], loss_fn, a0=0.5, c0=0.1, steps=config.SPSA_STEPS)
    final_loss = loss_fn(refined)

    w, final_resp = optical_chip.response(refined)
    ideal = optical_chip._IDEAL_RESPONSE

    global CURRENT_VOLTAGES
    CURRENT_VOLTAGES = refined.copy()

    return {
        "voltages": refined.tolist(),
        "loss": final_loss,
        "wavelengths": w.tolist(),
        "response": final_resp.tolist(),
        "ideal": ideal.tolist(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
