import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bayes_optimization.bayes_optimizer import config
from bayes_optimization.bayes_optimizer.simulate import optical_chip


def main():
    parser = argparse.ArgumentParser(description="Measure spectrum at given voltages")
    parser.add_argument("--volts", nargs="*", type=float, help="Voltage list")
    parser.add_argument("--out", type=str, default=None, help="Output plot path")
    args = parser.parse_args()

    if args.volts:
        volts = np.array(args.volts, dtype=float)
    else:
        volts = np.zeros(config.NUM_CHANNELS)

    w, resp = optical_chip.response(volts)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(w, resp)
        plt.xlabel("Wavelength")
        plt.ylabel("Response")
        plt.title(f"Scan voltages: {volts}")
        plt.tight_layout()
        plt.savefig(out_path)
        print(f"[INFO] 图像已保存至 {out_path}")
    else:
        for wl, r in zip(w, resp):
            print(f"{wl:.4f}, {r:.4f}")


if __name__ == "__main__":
    main()
