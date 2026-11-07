"""Pasarela de pagos simulada (Capa 2 – Persona 3).  Ejecutar: uvicorn api.main:app --reload"""
from fastapi import FastAPI

from api.esquemas import Lote, Resultado, ResultadoLote, Transaccion

app = FastAPI(title="Pasarela de pagos – Equipo 2", version="0.1")


@app.get("/api/v1/salud")
def salud():
    return {"status": "ok", "modelo": "pendiente", "umbral": 0.5, "version": "0.1"}


@app.post("/api/v1/transacciones", response_model=Resultado)
def evaluar(transaccion: Transaccion):
    raise NotImplementedError("Persona 3: pendiente")


@app.post("/api/v1/transacciones/lote", response_model=ResultadoLote)
def evaluar_lote(lote: Lote):
    raise NotImplementedError("Persona 3: pendiente")
