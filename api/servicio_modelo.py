"""Servicio de inferencia (Capa 2 – Persona 3).

Carga UNA sola vez modelo_fraude.pth, escalador.pkl y config_inferencia.json.
Mientras no exista el .pth, usa un DetectorFraudeMLP con pesos aleatorios
para que la Capa 3 pueda integrarse desde el día uno.
"""
from pathlib import Path

import torch

from nucleo.modelo import DetectorFraudeMLP

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ARTEFACTOS = RAIZ / "artefactos"


class ServicioFraude:
    def __init__(self) -> None:
        self.modelo = DetectorFraudeMLP()
        self.umbral = 0.5
        self.escalador = None
        # TODO Persona 3: cargar state_dict, escalador y config si existen en artefactos/
        self.modelo.eval()

    def predecir_lote(self, matriz: torch.Tensor):
        """Un solo forward para toda la ráfaga, bajo no_grad. Devuelve probabilidades."""
        with torch.no_grad():
            return torch.sigmoid(self.modelo(matriz)).squeeze(1)
