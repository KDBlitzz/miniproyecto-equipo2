"""Arquitectura topológica (Capa 1B – Persona 2).

Entradas: V1..V28 + Amount (29; Time se descartó). Punto de partida acordado por el equipo (sección 2.2 del reparto):
    29 → Linear(64) → BatchNorm1d → ReLU → Dropout(0.3)
       → Linear(32) → BatchNorm1d → ReLU → Dropout(0.3)
       → Linear(1)  (logit; la sigmoide vive en la pérdida y en la inferencia)

La API (Capa 2) importa ESTA clase para hacer load_state_dict, así que cualquier cambio
de arquitectura debe ir acompañado de un nuevo modelo_fraude.pth y config_inferencia.json.
"""
import torch
import torch.nn as nn


class DetectorFraudeMLP(nn.Module):
    """Perceptrón multicapa para clasificación binaria de transacciones."""

    def __init__(self, dim_entrada: int = 29, dims_ocultas=(64, 32), p_dropout: float = 0.3) -> None:
        super().__init__()
        # Configuración: se exporta a config_inferencia.json para que la API reconstruya la clase
        self.config = {"dim_entrada": dim_entrada, "dims_ocultas": list(dims_ocultas), "p_dropout": p_dropout}
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

    def desglose_parametros(self) -> list:
        """(nombre, cantidad) por tensor entrenable: para la tabla del informe."""
        return [(n, p.numel()) for n, p in self.named_parameters() if p.requires_grad]

    def huella_memoria_bytes(self, tamano_lote: int = 512) -> dict:
        """Cálculo analítico en float32 (4 B) para la Justificación Paramétrica."""
        n = self.contar_parametros()
        n_activ = tamano_lote * (2 * sum(self.config["dims_ocultas"]) + 1)
        return {"parametros": n, "pesos_B": n * 4, "gradientes_B": n * 4,
                "estado_AdamW_B": n * 8, "total_entrenamiento_B": n * 16,
                "activaciones_lote_B": n_activ * 4}


if __name__ == "__main__":
    modelo = DetectorFraudeMLP()
    print(modelo)
    print("Parámetros entrenables:", modelo.contar_parametros())  # 4225 con 29→64→32→1
