
"""API de la pasarela simulada. Ejecutar: uvicorn api.main:app --reload"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.esquemas import Lote, Resultado, ResultadoLote, Transaccion
from api.servicio_modelo import ServicioFraude


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.servicio = ServicioFraude()  # Se carga solo al arrancar.
    yield
    del app.state.servicio


app = FastAPI(title="Pasarela de pagos – Equipo 2", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def entrada_invalida(request: Request, exc: RequestValidationError):
    # El 422 por defecto repite el valor recibido; si es NaN o infinito no se puede
    # escribir en JSON y FastAPI termina en 500. Se omite el valor y se deja el resto.
    errores = [{k: v for k, v in e.items() if k != "input"} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errores)})


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
