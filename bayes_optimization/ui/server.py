from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import numpy as np
import csv
import sys
import asyncio

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayes_optimization.bayes_optimizer import (
    config,
    models,
    acquisition,
    optimizer,
    spsa,
    hardware,
)
from bayes_optimization.bayes_optimizer import calibrator
import json
import time
from bayes_optimization.bayes_optimizer.simulate import optical_chip

app = FastAPI()

static_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=static_dir), name="static")

CURRENT_MODE = "mock"
CALIBRATION = None
CURRENT_VOLTAGES = np.zeros(config.NUM_CHANNELS)
MANUAL_VOLTAGES = np.zeros(config.NUM_CHANNELS)
HARDWARE_CONNECTED = True
WAVEFORM_SOURCE = str(optical_chip.DATA_FILE.name)


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/status")
def get_status():
    return {
        "mode": CURRENT_MODE,
        "connected": HARDWARE_CONNECTED,
        "num_channels": config.NUM_CHANNELS,
        "voltages": CURRENT_VOLTAGES.tolist(),
        "manual": MANUAL_VOLTAGES.tolist(),
        "waveform_source": WAVEFORM_SOURCE,
    }


@app.post("/set_mode")
def set_mode_endpoint(data: dict):
    global CURRENT_MODE, HARDWARE_CONNECTED
    mode = data.get("mode")
    if mode not in {"mock", "real"}:
        raise HTTPException(status_code=400, detail="invalid mode")
    CURRENT_MODE = mode
    HARDWARE_CONNECTED = hardware.set_mode(mode)
    return {"mode": CURRENT_MODE, "connected": HARDWARE_CONNECTED}


@app.post("/set_channels")
def set_channels(data: dict):
    num = int(data.get("num_channels", config.NUM_CHANNELS))
    config.NUM_CHANNELS = num
    from bayes_optimization.bayes_optimizer.hardware.mock_hardware import MockOSA
    MockOSA.current_volts = np.zeros(num)
    global CURRENT_VOLTAGES, MANUAL_VOLTAGES
    CURRENT_VOLTAGES = np.zeros(num)
    MANUAL_VOLTAGES = np.zeros(num)
    return {"num_channels": num}


@app.post("/upload_waveform")
async def upload_waveform(file: UploadFile = File(...)):
    data = await file.read()
    name = file.filename or "uploaded"
    ext = Path(name).suffix.lower()
    if ext != ".csv":
        raise HTTPException(status_code=400, detail="only csv supported")
    try:
        text = data.decode("utf-8-sig")
        reader = csv.reader(text.strip().splitlines())
        rows = list(reader)
        if len(rows) < 2:
            raise ValueError("not enough rows")
        wl = np.asarray(rows[0], dtype=float)
        resp = np.asarray(rows[1], dtype=float)
        if wl.size == 0 or resp.size == 0 or wl.size != resp.size:
            raise ValueError("bad file")
    except Exception:
        raise HTTPException(status_code=400, detail="bad file")

    optical_chip.set_target_waveform(wl, resp)
    global WAVEFORM_SOURCE
    WAVEFORM_SOURCE = name
    w, ideal = optical_chip.get_target_waveform()
    return {
        "points": len(wl),
        "source": name,
        "wavelengths": w.tolist(),
        "ideal": ideal.tolist(),
    }


@app.post("/calibrate")
def run_calibrate():
    global CALIBRATION
    if CURRENT_MODE == "real" and not HARDWARE_CONNECTED:
        raise HTTPException(status_code=400, detail="hardware not connected")
    J = calibrator.measure_jacobian()
    n, mat = calibrator.compress_modes(J)
    CALIBRATION = {"modes": n, "matrix": mat.tolist()}
    return {"modes": n}


@app.get("/calibrate_stream")
def run_calibrate_stream():
    if CURRENT_MODE == "real" and not HARDWARE_CONNECTED:
        raise HTTPException(status_code=400, detail="hardware not connected")

    def event_generator():
        try:
            for data in calibrator.measure_jacobian_stream():
                if "step" in data:
                    payload = {
                        "step": data["step"],
                        "wavelengths": data["wavelengths"].tolist(),
                        "response": data["response"].tolist(),
                        "ideal": data["ideal"].tolist(),
                        "base": data["base"],
                        "perturb": data["perturb"],
                        "loss": data["loss"],
                    }
                    yield "data:" + json.dumps(payload) + "\n\n"
                    time.sleep(0.5)
                elif "done" in data:
                    J = data["matrix"]
                    n, mat = calibrator.compress_modes(J)
                    global CALIBRATION
                    CALIBRATION = {"modes": n, "matrix": mat.tolist()}
                    payload = {"done": True, "modes": n, "matrix": mat.tolist()}
                    yield "data:" + json.dumps(payload) + "\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            return

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def loss_fn(volts: np.ndarray) -> float:
    if CURRENT_MODE == "real":
        if not HARDWARE_CONNECTED:
            raise ConnectionError("hardware not connected")
        hardware.apply(volts)
        w, resp = hardware.read_spectrum()
    else:
        w, resp = optical_chip.response(volts)
    return optical_chip.compute_loss(w, resp)


@app.post("/manual")
def manual_adjust(data: dict):
    volts = np.array(data.get("voltages", []), dtype=float)
    if CURRENT_MODE == "real" and not HARDWARE_CONNECTED:
        raise HTTPException(status_code=400, detail="hardware not connected")
    try:
        hardware.apply(volts)
        w, resp = hardware.read_spectrum()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    global MANUAL_VOLTAGES
    MANUAL_VOLTAGES = volts.copy()
    _, ideal = optical_chip.get_target_waveform()
    return {
        "wavelengths": w.tolist(),
        "response": resp.tolist(),
        "ideal": ideal.tolist(),
        "voltages": MANUAL_VOLTAGES.tolist(),
    }


@app.post("/optimize")
def run_optimize():
    if CURRENT_MODE == "real" and not HARDWARE_CONNECTED:
        raise HTTPException(status_code=400, detail="hardware not connected")
    num_ch = config.NUM_CHANNELS
    bounds = np.tile(config.V_RANGE, (num_ch, 1))
    start = np.full(num_ch, sum(config.V_RANGE) / 2)

    gp = models.GaussianProcess()
    bo = optimizer.BayesOptimizer(gp, acquisition.expected_improvement, bounds)

    try:
        bo_res = bo.optimize(start, loss_fn, steps=config.BO_MAX_STEPS)
        refined = spsa.spsa_refine(bo_res["best_x"], loss_fn, a0=0.5, c0=0.1, steps=config.SPSA_STEPS)
        final_loss = loss_fn(refined)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    if CURRENT_MODE == "real":
        try:
            hardware.apply(refined)
            w, final_resp = hardware.read_spectrum()
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        w, final_resp = optical_chip.response(refined)
    _, ideal = optical_chip.get_target_waveform()

    global CURRENT_VOLTAGES, MANUAL_VOLTAGES
    CURRENT_VOLTAGES = refined.copy()
    MANUAL_VOLTAGES = refined.copy()

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
