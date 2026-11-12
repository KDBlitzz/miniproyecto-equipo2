
"""API de la pasarela simulada. Ejecutar: uvicorn api.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.esquemas import Lote, Resultado, ResultadoLote, Transaccion
from api.servicio_modelo import ServicioFraude


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.servicio = ServicioFraude()  # Se carga solo al arrancar.
    yield
    del app.state.servicio


app = FastAPI(title="Pasarela de pagos – Equipo 2", version="1.0", lifespan=lifespan)


@app.get("/api/v1/salud")
def salud(request: Request):
    servicio = request.app.state.servicio
    return {
        "status": "ok",
        "modelo": "modelo_fraude.pth",
        "umbral": servicio.umbral,
        "version": "1.0",
    }


@app.post("/api/v1/transacciones", response_model=Resultado)
def evaluar(transaccion: Transaccion, request: Request):
    return request.app.state.servicio.predecir(transaccion)


@app.post("/api/v1/transacciones/lote", response_model=ResultadoLote)
def evaluar_lote(lote: Lote, request: Request):
    return request.app.state.servicio.procesar_lote(lote.transacciones)
