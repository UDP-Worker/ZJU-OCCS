import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bayes_optimization.bayes_optimizer import config, calibrator, models, acquisition, optimizer, spsa
from bayes_optimization.bayes_optimizer.simulate import optical_chip


def loss_fn(volts: np.ndarray) -> float:
    _, resp = optical_chip.response(volts)
    return float(np.mean((resp - optical_chip._IDEAL_RESPONSE) ** 2))


def main():
    parser = argparse.ArgumentParser(description="Run mock optimization loop")
    parser.add_argument("--mode", default="mock", choices=["mock"], help="Only mock mode supported")
    parser.add_argument("--bo_steps", type=int, default=config.BO_MAX_STEPS, help="BO iterations")
    parser.add_argument("--spsa_steps", type=int, default=config.SPSA_STEPS, help="SPSA iterations")
    parser.add_argument("--out", type=str, default="data/reports/opt_result.png", help="Output plot path")
    args = parser.parse_args()

    num_ch = config.NUM_CHANNELS
    bounds = np.tile(config.V_RANGE, (num_ch, 1))
    start = np.full(num_ch, sum(config.V_RANGE) / 2)

    gp = models.GaussianProcess()
    bo = optimizer.BayesOptimizer(gp, acquisition.expected_improvement, bounds)

    print("[INFO] 开始贝叶斯优化...")
    bo_res = bo.optimize(start, loss_fn, steps=args.bo_steps)
    print(f"[INFO] BO最优损失 {bo_res['best_y']:.6f} @ {bo_res['best_x']}")

    print("[INFO] SPSA微调...")
    refined = spsa.spsa_refine(bo_res["best_x"], loss_fn, a0=0.5, c0=0.1, steps=args.spsa_steps)
    final_loss = loss_fn(refined)
    print(f"[INFO] 最终损失 {final_loss:.6f} @ {refined}")

    w, final_resp = optical_chip.response(refined)
    ideal = optical_chip._IDEAL_RESPONSE

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(w, ideal, label="ideal wave")
    plt.plot(w, final_resp, label="optimal result")
    plt.xlabel("Wavelength")
    plt.ylabel("Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[INFO] 图像已保存至 {out_path}")


if __name__ == "__main__":
    main()
