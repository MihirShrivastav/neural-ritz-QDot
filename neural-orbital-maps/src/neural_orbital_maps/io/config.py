from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class MaterialConfig(BaseModel):
    name: str = "GaAs"
    m_eff: float = 0.067
    epsilon_r: float = 12.9
    L0_nm: float = 30.0

    @field_validator("m_eff", "epsilon_r", "L0_nm")
    @classmethod
    def positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("material parameters must be positive")
        return value


class DomainConfig(BaseModel):
    x_extent: float = 4.0
    y_extent: float = 4.0
    num_points: int = 48

    @field_validator("x_extent", "y_extent")
    @classmethod
    def positive_extent(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("domain extents must be positive")
        return value

    @field_validator("num_points")
    @classmethod
    def enough_points(cls, value: int) -> int:
        if value < 8:
            raise ValueError("domain.num_points must be >= 8")
        return value


class DoubleDotConfig(BaseModel):
    kind: Literal["biquadratic"] = "biquadratic"
    separation: float = 1.4
    barrier: float = 5.0
    y_confinement: float = 4.0
    detuning: float = 0.0

    @field_validator("separation", "barrier", "y_confinement")
    @classmethod
    def positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("potential shape parameters must be positive")
        return value


class SolverConfig(BaseModel):
    num_states: int = Field(4, alias="K")
    basis_size: int = Field(6, alias="M")
    network: Literal["siren", "mlp"] = "siren"
    hidden_dim: int = 48
    hidden_layers: int = 2
    envelope_alpha: float = 0.12

    @field_validator("num_states", "basis_size", "hidden_dim", "hidden_layers")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("solver integer parameters must be positive")
        return value

    @model_validator(mode="after")
    def valid_basis(self) -> "SolverConfig":
        if self.num_states > self.basis_size:
            raise ValueError("solver.K must be <= solver.M")
        if self.envelope_alpha < 0:
            raise ValueError("solver.envelope_alpha must be non-negative")
        return self


class TrainingConfig(BaseModel):
    steps: int = 200
    lr: float = 5e-4
    log_every: int = 25
    seed: int = 42
    dtype: Literal["float32", "float64"] = "float64"
    device: str = "cpu"
    early_stop_patience: int = 6
    early_stop_min_delta: float = 1e-5

    @field_validator("steps", "log_every", "early_stop_patience")
    @classmethod
    def positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("training integer parameters must be positive")
        return value

    @field_validator("lr")
    @classmethod
    def positive_lr(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("training.lr must be positive")
        return value


class CoulombConfig(BaseModel):
    strength: Literal["material", "zero"] | float = "material"
    softening: float = 0.05
    integration: Literal["direct"] = "direct"

    @field_validator("softening")
    @classmethod
    def non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("coulomb.softening must be non-negative")
        return value


class PairConfig(BaseModel):
    enabled: bool = True
    num_orbitals: int = 4
    sectors: list[Literal["singlet", "triplet"]] = Field(default_factory=lambda: ["singlet", "triplet"])
    coulomb: CoulombConfig = Field(default_factory=CoulombConfig)

    @field_validator("num_orbitals")
    @classmethod
    def enough_orbitals(cls, value: int) -> int:
        if value < 2:
            raise ValueError("pair.num_orbitals must be >= 2")
        return value


class RunConfig(BaseModel):
    experiment_name: str = "smoke_pair"
    results_root: str = "results"
    material: MaterialConfig = Field(default_factory=MaterialConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    potential: DoubleDotConfig = Field(default_factory=DoubleDotConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    pair: PairConfig = Field(default_factory=PairConfig)

    @model_validator(mode="after")
    def valid_pair(self) -> "RunConfig":
        if self.pair.enabled and self.pair.num_orbitals > self.solver.num_states:
            raise ValueError("pair.num_orbitals must be <= solver.K")
        return self


def load_config(path: str | Path) -> RunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return RunConfig.model_validate(data)


def save_config(config: RunConfig, path: str | Path) -> None:
    payload = config.model_dump(by_alias=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


class SweepAxisConfig(BaseModel):
    min: float
    max: float
    num: int

    @field_validator("num")
    @classmethod
    def positive_num(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sweep axis num must be positive")
        return value

    def values(self) -> list[float]:
        if self.num == 1:
            return [float(self.min)]
        step = (self.max - self.min) / (self.num - 1)
        return [float(self.min + i * step) for i in range(self.num)]


class ExchangeMapConfig(BaseModel):
    study_name: str = "exchange_map"
    results_root: str = "results"
    base_config: RunConfig = Field(default_factory=RunConfig)
    detuning: SweepAxisConfig
    barrier: SweepAxisConfig


def load_exchange_map_config(path: str | Path) -> ExchangeMapConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return ExchangeMapConfig.model_validate(data)
