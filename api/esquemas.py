"""Esquemas Pydantic del contrato de la API (Capa 2 – Persona 3)."""
from typing import List, Literal

from pydantic import BaseModel, Field


class Transaccion(BaseModel):
    id_transaccion: str
    monto: float = Field(ge=0)
    tiempo: float = Field(ge=0)
    caracteristicas: List[float] = Field(min_length=28, max_length=28)  # V1..V28


class Resultado(BaseModel):
    id_transaccion: str
    status: Literal["approved", "blocked"]
    probabilidad_fraude: float
    umbral: float
    latencia_ms: float


class Lote(BaseModel):
    transacciones: List[Transaccion]


class ResultadoLote(BaseModel):
    resultados: List[Resultado]
    latencia_total_ms: float
    procesadas: int
