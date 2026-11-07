"""Funciones de costo para el desbalance de clases (Capa 1B – Persona 2).

Se implementan tres variantes para compararlas en el informe:
1. bce_estandar()   : BCEWithLogitsLoss sin pesos → línea base que demuestra el problema.
2. bce_ponderada()  : BCEWithLogitsLoss(pos_weight = N_neg / N_pos) → penalización por pesos.
3. FocalLoss        : Lin et al. (2017), FL = -alpha * (1 - p_t)^gamma * log(p_t).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_estandar() -> nn.Module:
    return nn.BCEWithLogitsLoss()


def bce_ponderada(n_negativos: int, n_positivos: int) -> nn.Module:
    """Entropía cruzada con peso para la clase positiva (fraude)."""
    raise NotImplementedError("Persona 2: pendiente")


class FocalLoss(nn.Module):
    """Focal Loss binaria sobre logits (estable numéricamente)."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduccion: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduccion = reduccion

    def forward(self, logits: torch.Tensor, objetivos: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Persona 2: pendiente")
