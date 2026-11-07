"""Bucle de entrenamiento explícito y módulo de validación (Capa 1B – Persona 2).

Requisitos del enunciado que viven aquí:
- optimizer.zero_grad() → loss.backward() → optimizer.step()  (escrito a mano)
- modelo.eval() + with torch.no_grad() en validación
- weight_decay (L2) en el optimizador (AdamW); comparación contra SGD
- registro por época de loss train / val y métricas → artefactos/historial_entrenamiento.csv
- exportación de artefactos/modelo_fraude.pth y artefactos/config_inferencia.json
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Entrenador:
    def __init__(self, modelo: nn.Module, perdida: nn.Module, optimizador: torch.optim.Optimizer,
                 dispositivo: str = "cpu") -> None:
        self.modelo = modelo.to(dispositivo)
        self.perdida = perdida
        self.optimizador = optimizador
        self.dispositivo = dispositivo
        self.historial = {"epoca": [], "loss_train": [], "loss_val": []}

    def entrenar_epoca(self, loader: DataLoader) -> float:
        """Una pasada por el conjunto de entrenamiento. Devuelve la loss promedio."""
        raise NotImplementedError("Persona 2: pendiente")

    def validar(self, loader: DataLoader):
        """Evalúa sin afectar el grafo computacional. Devuelve (loss, probabilidades, etiquetas)."""
        raise NotImplementedError("Persona 2: pendiente")

    def entrenar(self, loader_train: DataLoader, loader_val: DataLoader, epocas: int = 30,
                 paciencia: int = 5):
        """Ciclo completo con early stopping. Devuelve el historial."""
        raise NotImplementedError("Persona 2: pendiente")

    def guardar(self, ruta_modelo, ruta_config, umbral: float) -> None:
        raise NotImplementedError("Persona 2: pendiente")


if __name__ == "__main__":
    # Punto de entrada: python -m nucleo.entrenamiento
    raise NotImplementedError("Persona 2: pendiente")
