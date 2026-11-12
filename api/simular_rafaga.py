
"""Prueba ráfagas de 100, 500 y 1000 transacciones contra FastAPI.

Ejecutar desde la raíz: python -m api.simular_rafaga
Reporta percentiles de LATENCIA HTTP POR SOLICITUD (cada solicitud es un lote).
"""
import asyncio
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO = RAIZ / "artefactos" / "muestra_test.csv"
URL = "http://127.0.0.1:8000/api/v1/transacciones/lote"
TAMANO_SUBLOTE = 25  # Varias solicitudes por ráfaga para calcular p50/p95.


def leer_transacciones() -> list[dict]:
    df = pd.read_csv(ARCHIVO)
    columnas = [f"V{i}" for i in range(1, 29)]
    datos = []
    for i, fila in df.iterrows():
        datos.append({
            "id_transaccion": str(fila.get("id_transaccion", f"TX-{i:06d}")),
            "monto": float(fila["Amount"]),
            "tiempo": float(fila.get("Time", 0)),
            "caracteristicas": [float(fila[c]) for c in columnas],
        })
    return datos


async def enviar_lote(cliente: httpx.AsyncClient, datos: list[dict]) -> tuple[float, int]:
    inicio = time.perf_counter()
    respuesta = await cliente.post(URL, json={"transacciones": datos})
    respuesta.raise_for_status()
    duracion_ms = (time.perf_counter() - inicio) * 1000
    procesadas = respuesta.json()["procesadas"]
    if procesadas != len(datos):
        raise ValueError("La API no procesó todas las transacciones enviadas")
    return duracion_ms, procesadas


async def main() -> None:
    transacciones = leer_transacciones()
    async with httpx.AsyncClient(timeout=60) as cliente:
        for cantidad in (100, 500, 1000):
            if len(transacciones) < cantidad:
                raise ValueError(f"La muestra necesita al menos {cantidad} transacciones")
            seleccion = transacciones[:cantidad]
            sublotes = [seleccion[i:i + TAMANO_SUBLOTE]
                         for i in range(0, cantidad, TAMANO_SUBLOTE)]
            inicio = time.perf_counter()
            medidas = await asyncio.gather(
                *(enviar_lote(cliente, sublote) for sublote in sublotes)
            )
            segundos = time.perf_counter() - inicio
            latencias = np.array([medida[0] for medida in medidas])
            procesadas = sum(medida[1] for medida in medidas)
            print(f"\nRáfaga: {cantidad} transacciones ({len(sublotes)} solicitudes HTTP)")
            print(f"p50 HTTP: {np.percentile(latencias, 50):.2f} ms")
            print(f"p95 HTTP: {np.percentile(latencias, 95):.2f} ms")
            print(f"Tiempo total: {segundos * 1000:.2f} ms")
            print(f"Throughput: {procesadas / segundos:.2f} transacciones/segundo")


if __name__ == "__main__":
    asyncio.run(main())
