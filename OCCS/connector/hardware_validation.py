"""Manual validation script for the real hardware stack.

Run this module after connecting the SiliconExtreme DAC and the AQ6370 OSA to
verify end-to-end communication.  The script applies voltages, reads back the
values reported by the controller, and captures an optical trace so that the
operator can confirm the instruments are reacting as expected.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Mapping, Any

import numpy as np

from OCCS.connector.real_hardware import HardwareUnavailableError, RealHardware
from OCCS.connector.hardware_config import (
    DEFAULT_CONFIG_PATH,
    load_real_hardware_config,
)


def _parse_float_list(text: str, expected: int | None = None) -> List[float]:
    values = [float(item.strip()) for item in text.split(',') if item.strip()]
    if expected is not None and len(values) != expected:
        raise ValueError(f"expected {expected} values, got {len(values)}")
    return values


def _parse_bounds(text: str, dac_size: int) -> List[Tuple[float, float]]:
    parts = _parse_float_list(text)
    if len(parts) == 2:
        low, high = parts
        return [(low, high) for _ in range(dac_size)]
    if len(parts) != 2 * dac_size:
        raise ValueError("voltage bounds must be 2 values or 2*n values for per-channel limits")
    bounds: List[Tuple[float, float]] = []
    for idx in range(dac_size):
        bounds.append((parts[2 * idx], parts[2 * idx + 1]))
    return bounds


def _build_wavelength_grid(args: argparse.Namespace) -> np.ndarray:
    if args.wavelength_csv:
        path = Path(args.wavelength_csv)
        data = np.loadtxt(path, delimiter=',', dtype=float)
        if data.ndim == 1:
            return np.asarray(data, dtype=float)
        if data.ndim == 2 and data.shape[1] >= 1:
            return np.asarray(data[:, 0], dtype=float)
        raise ValueError("CSV file must contain at least one wavelength column")
    start = float(args.wavelength_start)
    stop = float(args.wavelength_stop)
    points = int(args.points)
    if points < 2:
        raise ValueError("at least two wavelength points are required")
    return np.linspace(start, stop, points)


def _format_summary(trace: np.ndarray) -> str:
    return (
        f"min: {trace.min():.3f} dBm, max: {trace.max():.3f} dBm, "
        f"mean: {trace.mean():.3f} dBm"
    )


def _positive_sleep(delay: float) -> None:
    if delay <= 0.0:
        return
    time.sleep(delay)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OCCS real-hardware validation tool")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="path to hardware_config.json (defaults to connector directory)",
    )
    parser.add_argument("--dac-port", help="override DAC port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--osa-resource", help="override VISA resource, e.g. GPIB0::1::INSTR")
    parser.add_argument("--dac-size", type=int, help="override number of DAC channels")
    parser.add_argument(
        "--channels",
        help="comma separated DAC channel numbers (defaults to configuration value)",
    )
    parser.add_argument(
        "--bounds",
        help="voltage bounds override: two numbers for all channels, or 2*n numbers for per-channel limits",
    )
    parser.add_argument(
        "--initial-voltage",
        help="initial voltage vector, comma separated",
    )
    parser.add_argument(
        "--test-voltage",
        help="optional test voltage vector applied after the initial measurement",
    )
    parser.add_argument(
        "--settle",
        type=float,
        default=0.2,
        help="wait time in seconds after applying voltages (default 0.2)",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=None,
        help="override wavelength sample count when CSV not provided",
    )
    parser.add_argument(
        "--wavelength-start",
        type=float,
        default=None,
        help="override wavelength start (meters)",
    )
    parser.add_argument(
        "--wavelength-stop",
        type=float,
        default=None,
        help="override wavelength stop (meters)",
    )
    parser.add_argument(
        "--wavelength-csv",
        help="optional CSV file with wavelength values (first column)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    config_path = Path(args.config)
    base_config: Mapping[str, Any] = load_real_hardware_config(config_path)
    if not base_config:
        raise SystemExit(
            f"hardware config 未初始化，请编辑 {config_path} 的 real_hardware 部分"
        )

    config_data = {
        "enabled": True,
        "dac": dict(base_config.get("dac", {})),
        "osa": dict(base_config.get("osa", {})),
        "bounds": base_config.get("bounds"),
    }

    dac_cfg = config_data["dac"]
    osa_cfg = config_data["osa"]

    if args.dac_port:
        dac_cfg["port"] = args.dac_port
    if args.channels:
        dac_cfg["channels"] = [int(ch) for ch in args.channels.split(',') if ch.strip()]
    if args.osa_resource:
        osa_cfg["resource"] = args.osa_resource
    dac_size = args.dac_size
    if dac_size is None:
        channels = dac_cfg.get("channels")
        dac_size = len(channels) if channels else None
    if not dac_size:
        raise SystemExit("无法确定 DAC 通道数量，请在配置文件或命令行中指定")

    if args.points is None:
        args.points = int(base_config.get("points", 801))
    if args.wavelength_start is None:
        args.wavelength_start = float(base_config.get("wavelength_start", 1.548e-6))
    if args.wavelength_stop is None:
        args.wavelength_stop = float(base_config.get("wavelength_stop", 1.552e-6))

    wavelength = _build_wavelength_grid(args)

    bounds = config_data.get("bounds")
    if args.bounds:
        bounds = _parse_bounds(args.bounds, dac_size)
    if bounds is not None:
        config_data["bounds"] = bounds

    channel_ids = dac_cfg.get("channels")
    if args.channels:
        channel_ids = [int(ch) for ch in args.channels.split(',') if ch.strip()]

    kwargs = {
        "config": config_data,
    }
    if channel_ids is not None:
        kwargs["channel_ids"] = channel_ids

    try:
        hardware = RealHardware(
            dac_size=dac_size,
            wavelength=wavelength,
            voltage_bounds=bounds,
            **kwargs,
        )
    except HardwareUnavailableError as exc:
        print(f"failed to initialise hardware: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - CLI error handling
        print(f"unexpected error: {exc}", file=sys.stderr)
        return 3

    try:
        initial = np.zeros(dac_size, dtype=float)
        if args.initial_voltage:
            initial = np.asarray(_parse_float_list(args.initial_voltage, dac_size), dtype=float)

        print("applying initial voltages:", initial)
        hardware.apply_voltage(initial)
        _positive_sleep(args.settle)
        readback = hardware.read_voltage()
        print("controller readback:", readback)

        trace = hardware.get_response()
        print("initial spectrum summary:")
        print(_format_summary(trace))

        if args.test_voltage:
            test_vec = np.asarray(_parse_float_list(args.test_voltage, dac_size), dtype=float)
            print("applying test voltages:", test_vec)
            hardware.apply_voltage(test_vec)
            _positive_sleep(args.settle)
            readback = hardware.read_voltage()
            print("controller readback during test:", readback)
            trace = hardware.get_response()
            print("test spectrum summary:")
            print(_format_summary(trace))

        print("validation complete — confirm the instruments respond as expected.")
        return 0
    finally:
        hardware.close()


if __name__ == "__main__":  # pragma: no cover - 脚本入口
    raise SystemExit(main())
