
"""Esquemas del contrato HTTP entre el dashboard y la API."""
from typing import Literal

from pydantic import BaseModel, Field


class Transaccion(BaseModel):
    id_transaccion: str = Field(min_length=1)
    monto: float = Field(ge=0, allow_inf_nan=False)
    tiempo: float = Field(ge=0, allow_inf_nan=False)
    caracteristicas: list[float] = Field(min_length=28, max_length=28)


class Resultado(BaseModel):
    id_transaccion: str
    status: Literal["approved", "blocked"]
    probabilidad_fraude: float
    umbral: float
    latencia_ms: float


class Lote(BaseModel):
    transacciones: list[Transaccion] = Field(min_length=1)


class ResultadoLote(BaseModel):
    resultados: list[Resultado]
    latencia_total_ms: float
    procesadas: int
