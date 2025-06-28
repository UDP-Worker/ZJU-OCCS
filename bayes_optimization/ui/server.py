from fastapi import FastAPI, Body
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
from bayes_optimization.bayes_optimizer.simulate import optical_chip

app = FastAPI()

static_dir = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=static_dir/"frontend"/"out"), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir/"frontend"/"out"/"index.html")


@app.get("/config")
def get_config():
    return {"num_channels": config.NUM_CHANNELS, "v_range": config.V_RANGE}


def loss_fn(volts: np.ndarray) -> float:
    _, resp = optical_chip.response(volts)
    return float(np.mean((resp - optical_chip._IDEAL_RESPONSE) ** 2))


@app.post("/simulate")
def simulate(volts: list[float] = Body(...)):
    arr = np.array(volts, dtype=float)
    w, resp = optical_chip.response(arr)
    return {
        "wavelengths": w.tolist(),
        "response": resp.tolist(),
        "ideal": optical_chip._IDEAL_RESPONSE.tolist(),
    }


@app.post("/optimize")
def run_optimize(mode: str = Body("mock")):
    # mode is ignored for now as we only implement mock hardware
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
