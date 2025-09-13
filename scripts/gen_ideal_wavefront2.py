import numpy as np
from pathlib import Path

try:
    # The package exposes get_response at OCCS.simulate
    from OCCS.simulate import get_response
except Exception:
    # fall back to the implementation path if package import hooks differ
    from OCCS.simulate.model import get_response  # type: ignore


def main() -> None:
    lam = np.linspace(1.55e-6, 1.56e-6, 200)
    true_volts = np.array([
        -0.8, -0.6, -0.4, -0.2,
         0.2,  0.4,  0.6,  0.8,
    ], dtype=float)
    target = get_response(lam, true_volts)

    out_dir = Path('OCCS') / 'data'
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'ideal_wavefront2.csv'
    # Two-row CSV: first row wavelengths, second row target values
    with out.open('w', encoding='utf-8') as f:
        f.write(','.join(f'{v:.16e}' for v in lam))
        f.write('\n')
        f.write(','.join(f'{v:.16e}' for v in target))
        f.write('\n')
    print(f'Wrote {out.resolve()} (shape={lam.size})')


if __name__ == '__main__':
    main()

