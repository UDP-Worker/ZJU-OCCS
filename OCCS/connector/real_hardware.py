"""Real hardware integration for the DAC/OSA laboratory setup.

This module mirrors the behaviour of the MATLAB utilities that control the
SiliconExtreme multi-channel voltage source and the Yokogawa AQ6370 optical
spectrum analyser.  The implementation keeps the public API identical to the
mock hardware so the optimiser can switch between them transparently.

The class is configurable via keyword arguments or environment variables so
that deployments can adapt to different instrument addresses without code
changes.  Critical imports (``pyvisa`` and ``pyserial``) are performed lazily to
remain importable in environments where the hardware stack is not installed.
"""

from __future__ import annotations

import atexit
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, List, Any

import numpy as np

from OCCS.connector.hardware_config import (
    DEFAULT_CONFIG_PATH,
    load_real_hardware_config,
)

logger = logging.getLogger(__name__)


class HardwareUnavailableError(RuntimeError):
    """Raised when the hardware resources cannot be initialised."""


@dataclass
class _OSASettings:
    resource: str
    channel: str
    sensitivity: str
    speed: str
    timeout_s: float
    chunk_size: int


@dataclass
class _DACSettings:
    port: str
    baudrate: int
    timeout_s: float
    write_timeout_s: float
    query_delay_s: float
    read_termination: str
    write_termination: str


def _normalise_bounds(
    bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]],
    size: int,
) -> Optional[List[Tuple[float, float]]]:
    if bounds is None:
        return None
    if isinstance(bounds, tuple) and len(bounds) == 2 and not any(
        isinstance(b, (list, tuple)) and len(b) == 2  # type: ignore[truthy-bool]
        for b in bounds
    ):
        low, high = float(bounds[0]), float(bounds[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError("Invalid voltage bounds: expected finite low < high")
        return [(low, high) for _ in range(size)]
    try:
        seq = list(bounds)  # type: ignore[arg-type]
    except TypeError as exc:  # not iterable
        raise ValueError("voltage_bounds must be (low, high) or a list of them") from exc
    if len(seq) != size:
        raise ValueError(
            f"voltage_bounds length mismatch: expected {size}, got {len(seq)}"
        )
    normalised: List[Tuple[float, float]] = []
    for idx, pair in enumerate(seq):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"voltage_bounds[{idx}] must be a (low, high) pair")
        low, high = float(pair[0]), float(pair[1])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError(
                f"Invalid bounds at index {idx}: expected finite low < high"
            )
        normalised.append((low, high))
    return normalised


class _SiliconExtremeController:
    """Lightweight wrapper around the SiliconExtreme serial protocol."""

    def __init__(self, settings: _DACSettings, serial_cls: Optional[Any] = None) -> None:
        try:
            if serial_cls is None:
                import serial  # type: ignore

                serial_cls = serial.Serial  # pragma: no cover - import side effect
        except ImportError as exc:  # pragma: no cover - requires optional dependency
            raise HardwareUnavailableError("pyserial is required for real hardware") from exc

        self._serial = serial_cls(
            port=settings.port,
            baudrate=settings.baudrate,
            parity='N',
            stopbits=1,
            bytesize=8,
            timeout=settings.timeout_s,
            write_timeout=settings.write_timeout_s,
        )
        self._settings = settings
        self._ensure_open()

    def _ensure_open(self) -> None:
        if not getattr(self._serial, "is_open", True):
            self._serial.open()

    def close(self) -> None:
        if getattr(self._serial, "is_open", False):
            self._serial.close()

    def _format_command(self, command: str) -> bytes:
        term = self._settings.write_termination
        if not command.endswith(term):
            command = f"{command}{term}"
        return command.encode("ascii")

    def _readline(self) -> str:
        raw = self._serial.readline()
        if not raw:
            return ""
        text = raw.decode("ascii", errors="ignore")
        return text.strip(self._settings.read_termination + " \r\n")

    def query(self, command: str) -> str:
        payload = self._format_command(command)
        self._serial.write(payload)
        self._serial.flush()
        if self._settings.query_delay_s:
            time.sleep(self._settings.query_delay_s)
        return self._readline()

    def set_voltage(self, channel: int, value: float) -> None:
        response = self.query(f"V{channel}={value:.4f}")
        if response.upper() != "OK":
            raise HardwareUnavailableError(
                f"Unexpected response when setting V{channel}: {response!r}"
            )

    def get_voltage(self, channel: int) -> float:
        response = self.query(f"V{channel}?")
        try:
            return float(response)
        except ValueError as exc:
            raise HardwareUnavailableError(
                f"Invalid voltage response for channel {channel}: {response!r}"
            ) from exc

    def set_vmax(self, channel: int, value: float) -> None:
        response = self.query(f"VMAX{channel}={value:.4f}")
        if response.upper() != "OK":
            raise HardwareUnavailableError(
                f"Failed to set VMAX{channel}: {response!r}"
            )

    def set_imax(self, channel: int, value: float) -> None:
        response = self.query(f"IMAX{channel}={value:.4f}")
        if response.upper() != "OK":
            raise HardwareUnavailableError(
                f"Failed to set IMAX{channel}: {response!r}"
            )


class _OSAController:
    """Wrapper around the Yokogawa AQ6370 command set via PyVISA."""

    def __init__(
        self,
        settings: _OSASettings,
        resource_manager: Optional[Any] = None,
    ) -> None:
        try:
            if resource_manager is None:
                import pyvisa  # type: ignore

                resource_manager = pyvisa.ResourceManager()  # pragma: no cover
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise HardwareUnavailableError("pyvisa is required for real hardware") from exc

        self._settings = settings
        self._rm = resource_manager
        self._instrument = self._rm.open_resource(settings.resource)
        self._instrument.timeout = int(settings.timeout_s * 1000)
        self._instrument.chunk_size = settings.chunk_size
        self._instrument.write_termination = "\n"
        self._instrument.read_termination = "\n"

    def close(self) -> None:
        self._instrument.close()
        # Some resource managers need explicit close
        if hasattr(self._rm, "close"):
            try:
                self._rm.close()
            except AttributeError:
                pass

    def _write(self, command: str) -> None:
        self._instrument.write(command)

    def _query(self, command: str) -> str:
        return self._instrument.query(command)

    def acquire_trace(
        self,
        start_nm: float,
        stop_nm: float,
        points: int,
        resolution_nm: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        channel = self._settings.channel
        self._write(f":sens:wav:star {start_nm}nm")
        self._write(f":sens:wav:stop {stop_nm}nm")
        self._write(f":sens:swe:points {points}")
        self._write(f":sens:band:res {resolution_nm}nm")
        self._write(f":sens:sens {self._settings.sensitivity}")
        self._write(f":trac:attr:tr{channel} write")
        self._write(":initiate:smode single")
        self._write(f":sens:swe:spe {self._settings.speed}")
        self._write(f":trac:stat:tr{channel} on")
        self._write("*CLS")
        self._write(":init")
        self._query("*OPC?")
        self._write(":format:data ascii")
        power = self._query(f":trac:y? tr{channel}")
        lam = self._query(f":trac:x? tr{channel}")
        power_arr = np.fromstring(power, sep=',', dtype=float)
        lam_arr = np.fromstring(lam, sep=',', dtype=float)
        if power_arr.size == 0 or lam_arr.size == 0:
            raise HardwareUnavailableError("OSA returned no data")
        return lam_arr, power_arr


class RealHardware:
    """Interface to the DAC/OSA hardware stack used in the laboratory."""

    def __init__(
        self,
        dac_size: int,
        wavelength: Iterable[float],
        voltage_bounds: Optional[Sequence[Tuple[float, float]] | Tuple[float, float]] = None,
        *,
        config: Optional[Mapping[str, Any]] = None,
        config_path: Optional[str | Path] = None,
        channel_ids: Optional[Sequence[int]] = None,
        dac_controller: Optional[Any] = None,
        osa_controller: Optional[Any] = None,
        serial_cls: Optional[Any] = None,
        visa_resource_manager: Optional[Any] = None,
    ) -> None:
        self.dac_size = int(dac_size)
        if self.dac_size <= 0:
            raise ValueError("dac_size must be positive")

        self.wavelength = np.asarray(wavelength, dtype=float)
        if self.wavelength.ndim != 1 or self.wavelength.size == 0:
            raise ValueError("wavelength must be a 1-D array with at least one sample")
        if not np.all(np.diff(self.wavelength) >= 0.0):
            raise ValueError("wavelength must be sorted in ascending order")

        config_data = self._load_effective_config(config, config_path)
        dac_config = dict(config_data.get("dac", {}))
        osa_config = dict(config_data.get("osa", {}))

        if voltage_bounds is None:
            voltage_bounds = self._extract_bounds_from_config(config_data)
        self.voltage_bounds = _normalise_bounds(voltage_bounds, self.dac_size)
        self._current_volts = np.zeros(self.dac_size, dtype=float)

        self._channel_ids = self._resolve_channel_ids(
            channel_ids,
            dac_config.get("channels"),
        )

        if osa_controller is None:
            osa_settings = self._build_osa_settings(osa_config)
            osa_controller = _OSAController(
                osa_settings,
                resource_manager=visa_resource_manager,
            )
        else:
            osa_settings = None

        if dac_controller is None:
            dac_settings = self._build_dac_settings(dac_config)
            dac_controller = _SiliconExtremeController(
                dac_settings,
                serial_cls=serial_cls,
            )
        else:
            dac_settings = None

        self._osa = osa_controller
        self._dac = dac_controller
        self._osa_settings = osa_settings
        self._dac_settings = dac_settings
        self._resolution_nm = (self.wavelength[-1] - self.wavelength[0]) * 1e9 / max(
            self.wavelength.size - 1, 1
        )
        self._closed = False

        # Apply hardware voltage limits when possible.
        if self.voltage_bounds is not None:
            for channel, (_, high) in zip(self._channel_ids, self.voltage_bounds):
                setter = getattr(self._dac, "set_vmax", None)
                if setter is None:
                    continue
                try:
                    setter(channel, high)
                except HardwareUnavailableError:
                    logger.warning("Unable to set VMAX for channel %s", channel)
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Unexpected error while setting VMAX for channel %s", channel)

        atexit.register(self.close)

    def _extract_bounds_from_config(
        self,
        config: Optional[Mapping[str, Any]],
    ) -> Optional[Sequence[Tuple[float, float]] | Tuple[float, float]]:
        if not config:
            return None
        bounds = config.get("bounds")
        if bounds is None:
            return None
        if isinstance(bounds, (list, tuple)):
            if len(bounds) == 2 and not any(isinstance(b, (list, tuple)) for b in bounds):
                return (float(bounds[0]), float(bounds[1]))
            converted: List[Tuple[float, float]] = []
            for idx, pair in enumerate(bounds):
                if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                    raise ValueError(
                        "Bounds in config must be either [low, high] or list of [low, high] pairs"
                    )
                converted.append((float(pair[0]), float(pair[1])))
            return converted
        raise ValueError("Bounds in config must be a list or tuple")

    def _load_effective_config(
        self,
        config: Optional[Mapping[str, Any]],
        config_path: Optional[str | Path],
    ) -> Mapping[str, Any]:
        if config is not None:
            return config
        base = load_real_hardware_config(config_path)
        if not base:
            raise HardwareUnavailableError(
                "Real hardware configuration is missing. Edit "
                f"{config_path or DEFAULT_CONFIG_PATH} before enabling the backend."
            )
        return base

    def _resolve_channel_ids(
        self,
        explicit: Optional[Sequence[int]],
        from_config: Optional[Sequence[int]],
    ) -> List[int]:
        if explicit is None:
            explicit = from_config
        if explicit is None:
            return list(range(1, self.dac_size + 1))
        if len(explicit) != self.dac_size:
            raise ValueError(
                f"channel_ids must match dac_size ({self.dac_size}), got {len(explicit)}"
        )
        return [int(ch) for ch in explicit]

    def _build_osa_settings(self, options: Mapping[str, Any]) -> _OSASettings:
        resource = options.get("resource")
        if not resource:
            raise HardwareUnavailableError(
                "OSA resource not configured. Update the hardware_config.json file."
            )
        channel = str(options.get("channel", "a")).strip()
        sensitivity = str(options.get("sensitivity", "high2")).strip()
        speed = str(options.get("speed", "2x")).strip()
        timeout_s = float(options.get("timeout", 150.0))
        chunk_size = int(options.get("chunk_size", 180009 * 3))
        return _OSASettings(
            resource=resource,
            channel=channel,
            sensitivity=sensitivity,
            speed=speed,
            timeout_s=timeout_s,
            chunk_size=chunk_size,
        )

    def _build_dac_settings(self, options: Mapping[str, Any]) -> _DACSettings:
        port = options.get("port")
        if not port:
            raise HardwareUnavailableError(
                "DAC port not configured. Update the hardware_config.json file."
            )
        baudrate = int(options.get("baudrate", 115200))
        timeout_s = float(options.get("timeout", 1.0))
        write_timeout_s = float(options.get("write_timeout", timeout_s))
        query_delay_s = float(options.get("query_delay", 0.05))
        read_termination = str(options.get("read_termination", "\n"))
        write_termination = str(options.get("write_termination", "\n"))
        return _DACSettings(
            port=port,
            baudrate=baudrate,
            timeout_s=timeout_s,
            write_timeout_s=write_timeout_s,
            query_delay_s=query_delay_s,
            read_termination=read_termination,
            write_termination=write_termination,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._dac.close()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to close DAC controller")
        try:
            self._osa.close()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to close OSA controller")

    def apply_voltage(self, new_volts: Iterable[float]) -> None:
        volts = np.asarray(new_volts, dtype=float)
        if volts.shape != (self.dac_size,):
            raise ValueError(
                f"Expected {self.dac_size} voltage values, got shape {volts.shape}"
            )
        if self.voltage_bounds is not None:
            lows = np.array([lo for lo, _ in self.voltage_bounds], dtype=float)
            highs = np.array([hi for _, hi in self.voltage_bounds], dtype=float)
            clipped = np.clip(volts, lows, highs)
            if not np.allclose(clipped, volts):
                logger.debug("Voltage vector clipped to hardware bounds")
            volts = clipped
        for channel, value in zip(self._channel_ids, volts):
            self._dac.set_voltage(int(channel), float(value))
        self._current_volts = volts

    def read_voltage(self) -> np.ndarray:
        readings: List[float] = []
        for idx, channel in enumerate(self._channel_ids):
            try:
                readings.append(self._dac.get_voltage(int(channel)))
            except HardwareUnavailableError:
                logger.warning("Falling back to cached voltage for channel %s", channel)
                readings.append(float(self._current_volts[idx]))
        return np.asarray(readings, dtype=float)

    def get_response(self) -> np.ndarray:
        start_nm = float(self.wavelength[0] * 1e9)
        stop_nm = float(self.wavelength[-1] * 1e9)
        points = int(self.wavelength.size)
        resolution_nm = max(self._resolution_nm, 1e-9)
        lambda_nm, power = self._osa.acquire_trace(start_nm, stop_nm, points, resolution_nm)
        lambda_m = lambda_nm * 1e-9
        if lambda_m.size != points or not np.allclose(lambda_m, self.wavelength, atol=1e-12):
            logger.debug("Interpolating OSA trace onto requested wavelength grid")
            power = np.interp(self.wavelength, lambda_m, power)
        return np.asarray(power, dtype=float)

    @property
    def skopt_dimensions(self) -> Optional[List[Tuple[float, float]]]:
        return self.voltage_bounds

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = ["RealHardware", "HardwareUnavailableError"]
