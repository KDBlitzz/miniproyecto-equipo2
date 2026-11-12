
"""Inferencia de fraude con los artefactos entrenados por la Capa 1."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from api.esquemas import Resultado, ResultadoLote, Transaccion
from nucleo.datos import Escalador
from nucleo.modelo import DetectorFraudeMLP

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ARTEFACTOS = RAIZ / "artefactos"


class ServicioFraude:
    def __init__(self) -> None:
        # Se cargan UNA sola vez, al iniciar FastAPI.
        ruta_config = RUTA_ARTEFACTOS / "config_inferencia.json"
        self.config = json.loads(ruta_config.read_text(encoding="utf-8"))
        self.columnas = self.config["columnas"]
        columnas_esperadas = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if self.columnas != columnas_esperadas:
            raise ValueError("Orden de columnas incompatible con la API")

        self.escalador = Escalador.cargar(RUTA_ARTEFACTOS / "escalador.pkl")
        if list(self.escalador.columnas) != self.columnas:
            raise ValueError("El escalador y la configuración usan distintas columnas")

        self.modelo = DetectorFraudeMLP(
            dim_entrada=self.config["dim_entrada"],
            dims_ocultas=tuple(self.config["dims_ocultas"]),
            p_dropout=self.config["p_dropout"],
        )
        pesos = torch.load(
            RUTA_ARTEFACTOS / "modelo_fraude.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.modelo.load_state_dict(pesos)
        self.modelo.eval()
        torch.set_num_threads(1)
        self.umbral = float(self.config["umbral"])

    def _preparar_matriz(self, transacciones: list[Transaccion]) -> torch.Tensor:
        # Una fila por transacción: V1...V28 y Amount al final.
        filas = [t.caracteristicas + [t.monto] for t in transacciones]
        matriz = np.asarray(filas, dtype=np.float32)
        matriz = self.escalador.transformar(matriz)
        if not np.isfinite(matriz).all():
            raise ValueError("La transacción contiene valores no finitos")
        return torch.as_tensor(matriz, dtype=torch.float32)

    def predecir_lote(self, matriz: torch.Tensor) -> list[float]:
        # Un único forward para todo el tensor; no se calculan gradientes.
        if matriz.shape[0] == 0:
            return []
        with torch.no_grad():
            probabilidades = torch.sigmoid(self.modelo(matriz)).flatten()
        return probabilidades.tolist()

    def predecir(self, transaccion: Transaccion) -> Resultado:
        inicio = time.perf_counter()
        probabilidad = self.predecir_lote(self._preparar_matriz([transaccion]))[0]
        latencia = (time.perf_counter() - inicio) * 1000
        return Resultado(
            id_transaccion=transaccion.id_transaccion,
            status="blocked" if probabilidad >= self.umbral else "approved",
            probabilidad_fraude=probabilidad,
            umbral=self.umbral,
            latencia_ms=latencia,
        )

    def procesar_lote(self, transacciones: list[Transaccion]) -> ResultadoLote:
        inicio = time.perf_counter()
        probabilidades = self.predecir_lote(self._preparar_matriz(transacciones))
        total_ms = (time.perf_counter() - inicio) * 1000
        # Cada elemento indica la latencia media imputada, no una medición individual.
        media_ms = total_ms / len(transacciones)
        resultados = [
            Resultado(
                id_transaccion=t.id_transaccion,
                status="blocked" if p >= self.umbral else "approved",
                probabilidad_fraude=p,
                umbral=self.umbral,
                latencia_ms=media_ms,
            )
            for t, p in zip(transacciones, probabilidades)
        ]
        return ResultadoLote(
            resultados=resultados,
            latencia_total_ms=total_ms,
            procesadas=len(resultados),
        )
