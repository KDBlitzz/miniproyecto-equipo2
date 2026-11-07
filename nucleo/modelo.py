"""Arquitectura topológica (Capa 1B – Persona 2).

Punto de partida acordado por el equipo (sección 2.2 del reparto):
    30 → Linear(64) → BatchNorm1d → ReLU → Dropout(0.3)
       → Linear(32) → BatchNorm1d → ReLU → Dropout(0.3)
       → Linear(1)  (logit; la sigmoide vive en la pérdida y en la inferencia)

La API (Capa 2) importa ESTA clase para hacer load_state_dict, así que cualquier cambio
de arquitectura debe ir acompañado de un nuevo modelo_fraude.pth y config_inferencia.json.
"""
import torch
import torch.nn as nn


class DetectorFraudeMLP(nn.Module):
    """Perceptrón multicapa para clasificación binaria de transacciones."""

    def __init__(self, dim_entrada: int = 30, dims_ocultas=(64, 32), p_dropout: float = 0.3) -> None:
        super().__init__()
        # __init__ contiene las capas
        self.capa1 = nn.Linear(dim_entrada, dims_ocultas[0])
        self.bn1 = nn.BatchNorm1d(dims_ocultas[0])
        self.capa2 = nn.Linear(dims_ocultas[0], dims_ocultas[1])
        self.bn2 = nn.BatchNorm1d(dims_ocultas[1])
        self.salida = nn.Linear(dims_ocultas[1], 1)
        self.activacion = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward define el flujo secuencial de los tensores
        x = self.dropout(self.activacion(self.bn1(self.capa1(x))))
        x = self.dropout(self.activacion(self.bn2(self.capa2(x))))
        return self.salida(x)  # logit de forma (batch, 1)

    def contar_parametros(self) -> int:
        """Recuento exacto de parámetros entrenables (debe coincidir con el cálculo a mano)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    modelo = DetectorFraudeMLP()
    print(modelo)
    print("Parámetros entrenables:", modelo.contar_parametros())  # 4289 con 30→64→32→1
