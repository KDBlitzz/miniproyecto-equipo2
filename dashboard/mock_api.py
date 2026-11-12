"""API mock (Persona 4): responde EXACTAMENTE el contrato 2.4 sin necesitar el modelo real.

Sirve para desarrollar el dashboard mientras la Capa 2 / el .pth no existen.
Ejecutar (desde la raíz del repo):

    uvicorn dashboard.mock_api:app --port 8000

Cuando Persona 3 tenga la API real, se apaga este mock y se levanta la de verdad
(`uvicorn api.main:app`): el dashboard no cambia porque el contrato es el mismo.

La "probabilidad" es una heurística, NO un modelo: en ULB los fraudes tienen V10, V12,
V14 y V17 muy negativas, así que se promedian y se pasan por una sigmoide.
"""
import math
import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

UMBRAL = 0.42
IDX_SENALES = (9, 11, 13, 16)  # V10, V12, V14, V17 (base 0 dentro de V1..V28)

app = FastAPI(title="Mock de la pasarela de pagos – Equipo 2", version="mock")


# --- Esquemas (copia del contrato; así el mock no depende de api/) -------------------------
class Transaccion(BaseModel):
    id_transaccion: str
    monto: float = Field(ge=0)
    tiempo: float = Field(ge=0)
    caracteristicas: List[float] = Field(min_length=28, max_length=28)


class Lote(BaseModel):
    transacciones: List[Transaccion]


# --- Lógica pura (el dashboard también la usa en el modo "Mock local") ---------------------
def puntuar(caracteristicas: List[float]) -> float:
    """Probabilidad de fraude simulada, en [0, 1]."""
    senal = -sum(caracteristicas[i] for i in IDX_SENALES) / len(IDX_SENALES)
    return 1.0 / (1.0 + math.exp(-1.2 * (senal - 3.0)))


def evaluar_una(tx: dict) -> dict:
    t0 = time.perf_counter()
    p = puntuar(tx["caracteristicas"])
    return {
        "id_transaccion": tx["id_transaccion"],
        "status": "blocked" if p >= UMBRAL else "approved",
        "probabilidad_fraude": round(p, 4),
        "umbral": UMBRAL,
        "latencia_ms": round((time.perf_counter() - t0) * 1000, 3),
    }


def evaluar_muchas(txs: List[dict]) -> dict:
    t0 = time.perf_counter()
    resultados = [evaluar_una(t) for t in txs]
    return {
        "resultados": resultados,
        "latencia_total_ms": round((time.perf_counter() - t0) * 1000, 3),
        "procesadas": len(resultados),
    }


# --- Endpoints del contrato ----------------------------------------------------------------
@app.get("/api/v1/salud")
def salud():
    return {"status": "ok", "modelo": "MOCK (sin modelo real)", "umbral": UMBRAL, "version": "mock"}


@app.post("/api/v1/transacciones")
def transaccion(tx: Transaccion):
    return evaluar_una(tx.model_dump())


@app.post("/api/v1/transacciones/lote")
def lote(lote: Lote):
    return evaluar_muchas([t.model_dump() for t in lote.transacciones])
