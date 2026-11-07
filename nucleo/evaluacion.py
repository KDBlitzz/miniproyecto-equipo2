"""Evaluación para clases desbalanceadas (Capa 1A – Persona 1).

La accuracy no sirve con 0.17 % de fraude. Se reportan:
precision, recall y F1 de la clase fraude, PR-AUC (average precision), ROC-AUC,
matriz de confusión y un barrido de umbral sobre validación para elegir el umbral final.
"""
import numpy as np


def calcular_metricas(y_real: np.ndarray, probabilidades: np.ndarray, umbral: float = 0.5) -> dict:
    """Devuelve un diccionario con todas las métricas para un umbral dado."""
    raise NotImplementedError("Persona 1: pendiente")


def barrido_umbral(y_real: np.ndarray, probabilidades: np.ndarray, umbrales=None):
    """Evalúa precision/recall/F1 para una rejilla de umbrales. Devuelve un DataFrame."""
    raise NotImplementedError("Persona 1: pendiente")


def elegir_umbral(y_real: np.ndarray, probabilidades: np.ndarray, criterio: str = "f1") -> float:
    """Elige el umbral que maximiza el criterio sobre el conjunto de validación."""
    raise NotImplementedError("Persona 1: pendiente")


def graficar_matriz_confusion(y_real, probabilidades, umbral, ruta_salida):
    raise NotImplementedError("Persona 1: pendiente")


def graficar_curva_pr(y_real, probabilidades, ruta_salida):
    raise NotImplementedError("Persona 1: pendiente")
