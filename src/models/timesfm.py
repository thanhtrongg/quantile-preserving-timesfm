"""Thin, version-aware adapter for the official TimesFM 2.5 PyTorch API.

The import of ``timesfm`` is deliberately lazy.  This keeps metric/storage
unit tests runnable without downloading a 200M-parameter checkpoint, while a
real baseline run fails with an actionable installation message.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# Verified against the official TimesFM 2.5 API in the pinned repository
# snapshot.  The first channel of quantile_forecast is the point/mean output;
# it is not exposed as a quantile by this adapter.
TIMESFM_25_QUANTILE_LEVELS = np.asarray(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64
)


@dataclass(frozen=True)
class TimesFMConfig:
    """Inference and precision settings for TimesFM 2.5."""

    checkpoint: str = "google/timesfm-2.5-200m-pytorch"
    dtype: str = "float32"
    context_length: int = 1024
    max_horizon: int = 256
    quantile_head: bool = True
    fix_quantile_crossing: bool = True
    normalize_inputs: bool = True
    force_flip_invariance: bool = True
    infer_is_positive: bool = False
    device: str = "auto"
    torch_compile: bool = True
    quantization_backend: str = "none"
    quantization_group_size: int = 128


def _as_numpy(value: Any) -> np.ndarray:
    """Convert NumPy/PyTorch/JAX-like values without retaining model tensors."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _as_input_list(history_batch: Sequence[Sequence[float]] | np.ndarray) -> list[np.ndarray]:
    if isinstance(history_batch, np.ndarray):
        values = np.asarray(history_batch, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise AssertionError(
                f"TimesFM expects a batch of univariate histories with shape (B, T), got {values.shape}"
            )
        histories = [row.copy() for row in values]
    else:
        histories = [np.asarray(history, dtype=np.float32).reshape(-1) for history in history_batch]
    if not histories or any(history.size == 0 for history in histories):
        raise ValueError("TimesFM cannot forecast from an empty history")
    return histories


def _resolve_quantile_levels(model: Any, quantile_count: int) -> np.ndarray:
    """Resolve levels from the model when available, otherwise use its API contract."""

    candidates = [
        getattr(model, "quantile_levels", None),
        getattr(getattr(model, "forecast_config", None), "quantile_levels", None),
        getattr(getattr(model, "_forecast_config", None), "quantile_levels", None),
    ]
    for candidate in candidates:
        if candidate is not None:
            levels = np.asarray(candidate, dtype=np.float64).reshape(-1)
            if levels.size == quantile_count:
                return levels

    if quantile_count == TIMESFM_25_QUANTILE_LEVELS.size:
        return TIMESFM_25_QUANTILE_LEVELS.copy()
    raise AssertionError(
        "TimesFM returned a quantile dimension that has no discoverable level metadata: "
        f"count={quantile_count}. Refuse to invent quantile levels."
    )


def normalize_forecast_output(
    point_forecast: Any,
    raw_quantile_forecast: Any,
    *,
    model: Any | None = None,
) -> dict[str, np.ndarray]:
    """Normalize the official tuple to the project forecast contract.

    Official TimesFM 2.5 returns ``(B, H)`` point forecasts and ``(B, H, 10)``
    quantile forecasts whose first channel is the mean/point forecast followed
    by q10 through q90.  The returned quantile tensor is therefore ``(B, H, 9)``.
    """

    point = _as_numpy(point_forecast).astype(np.float32, copy=False)
    raw = _as_numpy(raw_quantile_forecast).astype(np.float32, copy=False)
    if point.ndim != 2:
        raise AssertionError(f"point_forecast must have shape (B, H), got {point.shape}")
    if raw.ndim != 3:
        raise AssertionError(
            f"raw quantile forecast must have shape (B, H, Q+1), got {raw.shape}"
        )
    if raw.shape[:2] != point.shape:
        raise AssertionError(
            f"point and quantile forecast shapes disagree: {point.shape} vs {raw.shape}"
        )
    if raw.shape[-1] < 2:
        raise AssertionError("TimesFM quantile output must contain point plus at least one quantile")

    levels = _resolve_quantile_levels(model, raw.shape[-1] - 1)
    if levels.ndim != 1 or not np.all(np.diff(levels) > 0):
        raise AssertionError(f"Quantile levels must be strictly increasing, got {levels}")
    quantiles = raw[..., 1:]
    if quantiles.shape[-1] != levels.size:
        raise AssertionError("Quantile metadata does not match quantile forecast shape")
    return {
        "point_forecast": point,
        "quantile_forecasts": quantiles,
        "quantile_levels": levels,
    }


class TimesFMAdapter:
    """Adapter exposing a stable ``forecast(history_batch, prediction_length)`` API."""

    def __init__(self, config: TimesFMConfig, model: Any | None = None):
        if config.dtype.lower() not in {"float32", "fp32", "bfloat16", "bf16"}:
            raise ValueError(
                f"Unsupported TimesFM floating-point dtype={config.dtype!r}. "
                "Use float32 or bfloat16."
            )
        if config.quantization_backend not in {
            "none",
            "torchao_int8_weight_only",
            "torchao_int8_dynamic_activation_int8_weight",
            "torchao_int4_weight_only",
        }:
            raise ValueError(
                "Unsupported quantization backend: "
                f"{config.quantization_backend!r}"
            )
        if config.quantization_group_size <= 0:
            raise ValueError("quantization_group_size must be positive")
        self.config = config
        self._autocast_dtype: Any | None = None
        self._quantized_linear_names: list[str] = []
        self._excluded_quantization_module_types: list[str] = []
        self._model = model if model is not None else self._load_model()
        self._prepare_precision()
        self._compile_model()

    def _load_model(self) -> Any:
        try:
            import timesfm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "TimesFM 2.5 is not installed. Install the pinned runtime with "
                "`pip install -e .[runtime]` or `pip install -r requirements-lock.txt`."
            ) from exc

        model_class = getattr(timesfm, "TimesFM_2p5_200M_torch", None)
        if model_class is None:
            raise RuntimeError(
                "Installed `timesfm` does not expose the TimesFM 2.5 PyTorch API. "
                "Do not fall back to the archived TimesFM 1.x API."
            )
        # Quantization must happen before compiling the module.  The FP32/BF16
        # path can use the checkpoint loader's compiled forward directly.
        compile_on_load = self.config.torch_compile and self.config.quantization_backend == "none"
        model = model_class.from_pretrained(
            self.config.checkpoint,
            torch_compile=compile_on_load,
        )
        if self.config.device != "auto" and hasattr(model, "to"):
            model = model.to(self.config.device)
        return model

    def _model_module(self) -> Any:
        return getattr(self._model, "model", self._model)

    def _prepare_precision(self) -> None:
        """Apply an explicit BF16 or TorchAO PTQ workflow before compile()."""

        module = self._model_module()
        dtype = self.config.dtype.lower()
        if dtype in {"bfloat16", "bf16"}:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - runtime-only path
                raise RuntimeError("PyTorch is required for BF16 TimesFM") from exc
            module.to(torch.bfloat16)
            self._autocast_dtype = torch.bfloat16

        backend = self.config.quantization_backend
        if backend == "none":
            return
        try:
            from torchao.quantization import (
                Int4WeightOnlyConfig,
                Int8DynamicActivationInt8WeightConfig,
                Int8WeightOnlyConfig,
                quantize_,
            )
        except ImportError as exc:
            raise RuntimeError(
                "TorchAO is required for the requested real PTQ backend."
            ) from exc

        if backend == "torchao_int8_weight_only":
            quantize_(module, Int8WeightOnlyConfig())
        elif backend == "torchao_int8_dynamic_activation_int8_weight":
            quantize_(module, Int8DynamicActivationInt8WeightConfig())
        elif backend == "torchao_int4_weight_only":
            try:
                quantize_(
                    module,
                    Int4WeightOnlyConfig(
                        group_size=self.config.quantization_group_size
                    ),
                )
            except Exception as exc:
                raise RuntimeError(
                    "TorchAO INT4 weight-only quantization is unavailable for "
                    f"this TimesFM/device combination: {type(exc).__name__}: {exc}"
                ) from exc

        if self.config.torch_compile:
            try:
                import torch

                module.forward = torch.compile(module.forward)
            except Exception as exc:
                raise RuntimeError(
                    "The requested TorchAO quantized model could not be compiled: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        self._quantized_linear_names = [
            name
            for name, child in module.named_modules()
            if child.__class__.__name__ in {"Linear"}
            and hasattr(getattr(child, "weight", None), "qdata")
        ]
        self._excluded_quantization_module_types = sorted(
            {
                child.__class__.__name__
                for name, child in module.named_modules()
                if name and child.__class__.__name__ not in {"Linear"}
            }
        )

    def _compile_model(self) -> None:
        try:
            import timesfm  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("The pinned TimesFM package is required to compile the model") from exc

        forecast_config = timesfm.ForecastConfig(
            max_context=self.config.context_length,
            max_horizon=self.config.max_horizon,
            normalize_inputs=self.config.normalize_inputs,
            use_continuous_quantile_head=self.config.quantile_head,
            force_flip_invariance=self.config.force_flip_invariance,
            infer_is_positive=self.config.infer_is_positive,
            fix_quantile_crossing=self.config.fix_quantile_crossing,
        )
        if not hasattr(self._model, "compile"):
            raise RuntimeError("The loaded TimesFM 2.5 object has no compile(ForecastConfig) method")
        self._model.compile(forecast_config)
        self.forecast_config = forecast_config

    @property
    def model(self) -> Any:
        return self._model

    def forecast(
        self,
        history_batch: Sequence[Sequence[float]] | np.ndarray,
        prediction_length: int,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if prediction_length > self.config.max_horizon:
            raise ValueError(
                f"prediction_length={prediction_length} exceeds configured max_horizon="
                f"{self.config.max_horizon}"
            )
        inputs = _as_input_list(history_batch)
        if self.config.context_length:
            inputs = [history[-self.config.context_length :] for history in inputs]
        with self._autocast_context():
            point, raw_quantiles = self._model.forecast(
                horizon=prediction_length, inputs=inputs
            )
        normalized = normalize_forecast_output(point, raw_quantiles, model=self._model)
        expected_batch = len(inputs)
        if normalized["point_forecast"].shape != (expected_batch, prediction_length):
            raise AssertionError(
                "TimesFM returned an unexpected point forecast shape: "
                f"{normalized['point_forecast'].shape}; expected {(expected_batch, prediction_length)}"
            )
        if normalized["quantile_forecasts"].shape[:2] != (expected_batch, prediction_length):
            raise AssertionError(
                "TimesFM returned an unexpected quantile forecast shape: "
                f"{normalized['quantile_forecasts'].shape}"
            )
        return normalized

    def _autocast_context(self):
        if getattr(self, "_autocast_dtype", None) is None:
            return nullcontext()
        try:
            import torch

            module = self._model_module()
            device = next(module.parameters()).device
            if device.type not in {"cpu", "cuda"}:
                return nullcontext()
            return torch.autocast(device_type=device.type, dtype=self._autocast_dtype)
        except (ImportError, StopIteration, AttributeError):
            return nullcontext()

    def quantization_metadata(self) -> Mapping[str, Any]:
        module = self._model_module()
        linear_names = [
            name for name, child in module.named_modules() if child.__class__.__name__ == "Linear"
        ]
        return {
            "backend": self.config.quantization_backend,
            "dtype": self.config.dtype,
            "torch_compile": self.config.torch_compile,
            "quantized_linear_module_count": len(self._quantized_linear_names),
            "quantized_linear_modules": self._quantized_linear_names,
            "total_linear_module_count": len(linear_names),
            "excluded_module_types": self._excluded_quantization_module_types,
            "group_size": (
                self.config.quantization_group_size
                if "int4" in self.config.quantization_backend
                else None
            ),
            "scheme": {
                "torchao_int8_weight_only": "weight-only; symmetric per-row weight quantization; FP32 activations",
                "torchao_int8_dynamic_activation_int8_weight": "dynamic symmetric activations and per-row int8 weights",
                "torchao_int4_weight_only": "weight-only; groupwise int4; group_size=128; CPU compatibility check only",
                "none": "no quantization",
            }.get(self.config.quantization_backend, "unknown"),
        }

    def metadata(self) -> Mapping[str, Any]:
        try:
            import timesfm  # type: ignore[import-not-found]
            from importlib.metadata import version as package_version

            version = getattr(timesfm, "__version__", None) or package_version("timesfm")
        except ImportError:
            version = "not-installed"
        return {
            "library": "timesfm",
            "version": version,
            "checkpoint": self.config.checkpoint,
            "dtype": self.config.dtype,
            "context_length": self.config.context_length,
            "max_horizon": self.config.max_horizon,
            "quantile_head": self.config.quantile_head,
            "fix_quantile_crossing": self.config.fix_quantile_crossing,
            "torch_compile": self.config.torch_compile,
            "quantization_backend": self.config.quantization_backend,
        }
