"""Ingesta y preprocesamiento (Capa 1A – Persona 1).

Responsabilidades:
1. Descargar el dataset Credit Card Fraud (ULB) y guardarlo en data/creditcard.csv.
2. Análisis exploratorio del desbalance (conteos, porcentajes, distribución de Amount).
3. Partición estratificada train / val / test (70 / 15 / 15) con semilla 42.
4. Escalador Z-score ajustado SOLO con train y guardado en artefactos/escalador.pkl.
5. TransaccionesDataset(torch.utils.data.Dataset) y DataLoaders por mini-batches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Rutas relativas a la raíz del repositorio
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "creditcard.csv"
RUTA_ARTEFACTOS = RAIZ / "artefactos"
SEMILLA = 42

# Columnas del dataset: Time, V1..V28, Amount (entradas) y Class (etiqueta, 1 = fraude)
COLUMNAS_ENTRADA = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
COLUMNA_ETIQUETA = "Class"


def descargar_dataset(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Descarga el dataset desde OpenML (sin cuenta de Kaggle) y lo guarda como CSV.

    Si el CSV ya existe, solo lo lee.
    """
    raise NotImplementedError("Persona 1: pendiente")


def resumen_desbalance(df: pd.DataFrame) -> dict:
    """Devuelve conteo y porcentaje por clase, y estadísticas de Amount por clase."""
    raise NotImplementedError("Persona 1: pendiente")


def particionar(df: pd.DataFrame, semilla: int = SEMILLA):
    """Partición estratificada 70/15/15. Devuelve (X_train, y_train, X_val, y_val, X_test, y_test)."""
    raise NotImplementedError("Persona 1: pendiente")


class Escalador:
    """Estandarización Z-score: z = (x - media) / desviación.

    Se ajusta únicamente con el conjunto de entrenamiento y se guarda para que la API
    aplique exactamente la misma transformación en inferencia.
    """

    def __init__(self) -> None:
        self.media: np.ndarray | None = None
        self.desviacion: np.ndarray | None = None

    def ajustar(self, X: np.ndarray) -> "Escalador":
        raise NotImplementedError("Persona 1: pendiente")

    def transformar(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Persona 1: pendiente")

    def guardar(self, ruta: Path = RUTA_ARTEFACTOS / "escalador.pkl") -> None:
        raise NotImplementedError("Persona 1: pendiente")

    @classmethod
    def cargar(cls, ruta: Path = RUTA_ARTEFACTOS / "escalador.pkl") -> "Escalador":
        raise NotImplementedError("Persona 1: pendiente")


class TransaccionesDataset(Dataset):
    """Dataset personalizado: entrega (tensor de 30 características, etiqueta float)."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Persona 1: pendiente")

    def __len__(self) -> int:
        raise NotImplementedError("Persona 1: pendiente")

    def __getitem__(self, indice: int):
        raise NotImplementedError("Persona 1: pendiente")


def crear_dataloaders(tamano_lote: int = 512, semilla: int = SEMILLA):
    """Pipeline completo: descarga → partición → escalador → Datasets → DataLoaders.

    Devuelve (loader_train, loader_val, loader_test, escalador).
    shuffle=True solo en entrenamiento.
    """
    raise NotImplementedError("Persona 1: pendiente")
