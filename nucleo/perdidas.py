"""Funciones de costo para el desbalance de clases (Capa 1B – Persona 2).

1. bce_estandar()   : línea base (demuestra el problema).
2. bce_ponderada()  : BCEWithLogitsLoss(pos_weight = N_neg / N_pos).
3. FocalLoss        : Lin et al. (2017), FL = -alpha_t * (1 - p_t)^gamma * log(p_t).
Todas aceptan logits (B,) o (B,1) y etiquetas (B,) o (B,1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_estandar() -> nn.Module:
    return _BCE(None)


def bce_ponderada(n_negativos: int, n_positivos: int, suavizado: bool = False) -> nn.Module:
    """Entropía cruzada con peso para la clase positiva (fraude).
    suavizado=True usa raíz cuadrada del cociente (~24 en vez de ~578): más estable."""
    w = n_negativos / max(n_positivos, 1)
    if suavizado:
        w = w ** 0.5
    return _BCE(torch.tensor([w], dtype=torch.float32))


class _BCE(nn.Module):
    def __init__(self, pos_weight) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)  # viaja con .to(dispositivo)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, objetivos: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits.view(-1), objetivos.view(-1).float(), pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    """Focal Loss binaria sobre logits (estable numéricamente).

    alpha = peso de la clase fraude (la legítima recibe 1 - alpha). Con 0.17 % de fraude
    conviene alpha alto (0.75–0.99); 0.25 le daría MENOS peso al fraude.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduccion: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduccion = reduccion

    def forward(self, logits: torch.Tensor, objetivos: torch.Tensor) -> torch.Tensor:
        logits, y = logits.view(-1), objetivos.view(-1).float()
        bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")  # = -log(p_t)
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * y + (1.0 - self.alpha) * (1.0 - y)
        perdida = alpha_t * (1.0 - p_t) ** self.gamma * bce
        if self.reduccion == "mean":
            return perdida.mean()
        if self.reduccion == "sum":
            return perdida.sum()
        return perdida
