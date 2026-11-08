"""Bucle de entrenamiento explícito y módulo de validación (Capa 1B – Persona 2).

Uso:  python -m nucleo.entrenamiento --perdida ponderada --optimizador adamw --exportar
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score  # solo MÉTRICA; la red es PyTorch puro
from torch.utils.data import DataLoader

RAIZ = Path(__file__).resolve().parent.parent
RUTA_ARTEFACTOS = RAIZ / "artefactos"


def _metricas(probs: np.ndarray, y: np.ndarray, umbral: float) -> dict:
    pred = probs >= umbral
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12),
            "pr_auc": float(average_precision_score(y, probs)) if y.sum() > 0 else 0.0}


class Entrenador:
    def __init__(self, modelo: nn.Module, perdida: nn.Module, optimizador: torch.optim.Optimizer,
                 dispositivo: str = "cpu", umbral_metricas: float = 0.5) -> None:
        self.modelo = modelo.to(dispositivo)
        self.perdida = perdida.to(dispositivo)
        self.optimizador = optimizador
        self.dispositivo = dispositivo
        self.umbral_metricas = umbral_metricas
        self.historial = {"epoca": [], "loss_train": [], "loss_val": [],
                          "precision_val": [], "recall_val": [], "f1_val": [], "pr_auc_val": []}

    def entrenar_epoca(self, loader: DataLoader) -> float:
        """Una pasada por el conjunto de entrenamiento. Devuelve la loss promedio."""
        self.modelo.train()                       # activa Dropout y BatchNorm en modo entrenamiento
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(self.dispositivo), y.to(self.dispositivo).float()
            self.optimizador.zero_grad()          # 1) limpia los gradientes acumulados
            logits = self.modelo(x)               #    paso hacia adelante
            loss = self.perdida(logits, y)        #    costo (modificado para el desbalance)
            loss.backward()                       # 2) retropropagación del error
            self.optimizador.step()               # 3) actualización de pesos (AdamW incluye L2 desacoplado)
            total += loss.item() * len(y)
            n += len(y)
        return total / n

    def validar(self, loader: DataLoader):
        """Evalúa sin afectar el grafo computacional. Devuelve (loss, probabilidades, etiquetas)."""
        self.modelo.eval()                        # Dropout apagado; BN usa estadísticas acumuladas
        total, n, probs, ys = 0.0, 0, [], []
        with torch.no_grad():                     # no se construye grafo
            for x, y in loader:
                x, y = x.to(self.dispositivo), y.to(self.dispositivo).float()
                logits = self.modelo(x)
                total += self.perdida(logits, y).item() * len(y)
                n += len(y)
                probs.append(torch.sigmoid(logits).view(-1).cpu())
                ys.append(y.view(-1).cpu())
        return total / n, torch.cat(probs).numpy(), torch.cat(ys).numpy()

    def entrenar(self, loader_train: DataLoader, loader_val: DataLoader, epocas: int = 30,
                 paciencia: int = 5, verbose: bool = True):
        """Ciclo completo con early stopping por PR-AUC de validación. Devuelve el historial."""
        mejor, mejor_estado, sin_mejora = -1.0, None, 0
        for ep in range(1, epocas + 1):
            loss_tr = self.entrenar_epoca(loader_train)
            loss_va, probs, y = self.validar(loader_val)
            m = _metricas(probs, y, self.umbral_metricas)
            for k, v in zip(self.historial, [ep, loss_tr, loss_va, m["precision"], m["recall"],
                                             m["f1"], m["pr_auc"]]):
                self.historial[k].append(v)
            if verbose:
                print(f"Ep {ep:02d} | train {loss_tr:.5f} | val {loss_va:.5f} | "
                      f"P {m['precision']:.3f} R {m['recall']:.3f} PR-AUC {m['pr_auc']:.3f}")
            if m["pr_auc"] > mejor:
                mejor, sin_mejora = m["pr_auc"], 0
                mejor_estado = copy.deepcopy(self.modelo.state_dict())
            else:
                sin_mejora += 1
                if sin_mejora >= paciencia:
                    if verbose:
                        print(f"Early stopping en la época {ep} (mejor PR-AUC val = {mejor:.4f})")
                    break
        self.modelo.load_state_dict(mejor_estado)  # restaura la mejor versión
        return self.historial

    def guardar_historial(self, ruta_csv) -> None:
        pd.DataFrame(self.historial).to_csv(ruta_csv, index=False)

    def guardar(self, ruta_modelo, ruta_config, umbral: float, extra: dict = None) -> None:
        """modelo_fraude.pth = state_dict puro (contrato 2.5); config_inferencia.json aparte."""
        torch.save(self.modelo.state_dict(), ruta_modelo)
        config = dict(getattr(self.modelo, "config", {}))
        config["umbral"] = float(umbral)
        config.update(extra or {})
        with open(ruta_config, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def _umbral_f1(probs: np.ndarray, y: np.ndarray) -> float:
    """Respaldo local mientras evaluacion.elegir_umbral (Persona 1) no exista."""
    rejilla = np.linspace(0.05, 0.95, 91)
    return float(rejilla[int(np.argmax([_metricas(probs, y, t)["f1"] for t in rejilla]))])


def principal() -> None:
    from nucleo.datos import crear_dataloaders
    try:
        from nucleo.datos import COLUMNAS_ENTRADA
    except ImportError:
        COLUMNAS_ENTRADA = None
    from nucleo.modelo import DetectorFraudeMLP
    from nucleo.perdidas import FocalLoss, bce_estandar, bce_ponderada

    ap = argparse.ArgumentParser()
    ap.add_argument("--perdida", choices=["estandar", "ponderada", "ponderada_sqrt", "focal"], default="ponderada")
    ap.add_argument("--optimizador", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--lote", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--ocultas", type=int, nargs="+", default=[64, 32],
                    help="capas ocultas, p. ej. --ocultas 128 64  o  --ocultas 32 16")
    ap.add_argument("--criterio", choices=["f1", "costo"], default="f1",
                    help="cómo elegir el umbral sobre validación (evaluacion.elegir_umbral)")
    ap.add_argument("--exportar", action="store_true", help="escribe modelo_fraude.pth y config_inferencia.json")
    a = ap.parse_args()

    torch.manual_seed(42)
    datos = crear_dataloaders(a.lote)          # diccionario (ver docstring de nucleo/datos.py)
    loader_train, loader_val = datos["loader_train"], datos["loader_val"]
    n_pos = datos["dataset_train"].n_positivos
    n_neg = len(datos["dataset_train"]) - n_pos
    print(f"Train: {n_neg} legítimas, {n_pos} fraudes")

    perdida = {"estandar": bce_estandar,
               "ponderada": lambda: bce_ponderada(n_neg, n_pos),
               "ponderada_sqrt": lambda: bce_ponderada(n_neg, n_pos, suavizado=True),
               "focal": lambda: FocalLoss(a.alpha, a.gamma)}[a.perdida]()

    dim_entrada = next(iter(loader_train))[0].shape[1]   # 29 si Persona 1 quitó Time
    modelo = DetectorFraudeMLP(dim_entrada, tuple(a.ocultas))
    print("Parámetros entrenables:", modelo.contar_parametros())
    opt = (torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=1e-4) if a.optimizador == "adamw"
           else torch.optim.SGD(modelo.parameters(), lr=a.lr * 10, momentum=0.9, weight_decay=1e-4))

    ent = Entrenador(modelo, perdida, opt)
    ent.entrenar(loader_train, loader_val, epocas=a.epocas)

    RUTA_ARTEFACTOS.mkdir(exist_ok=True)
    ent.guardar_historial(RUTA_ARTEFACTOS / f"historial_{a.perdida}_{a.optimizador}_{'-'.join(map(str, a.ocultas))}.csv")

    _, probs, y = ent.validar(loader_val)
    try:
        from nucleo.evaluacion import elegir_umbral
        umbral, _ = elegir_umbral(y, probs, criterio=a.criterio)   # devuelve (umbral, métricas)
    except NotImplementedError:
        umbral = _umbral_f1(probs, y)
    m = _metricas(probs, y, umbral)
    print(f"Umbral {umbral:.2f} -> P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} PR-AUC {m['pr_auc']:.3f}")

    if a.exportar:
        ent.guardar(RUTA_ARTEFACTOS / "modelo_fraude.pth", RUTA_ARTEFACTOS / "config_inferencia.json",
                    umbral, extra={"columnas": COLUMNAS_ENTRADA, "perdida": a.perdida,
                                   "alpha": a.alpha, "gamma": a.gamma})
        ent.guardar_historial(RUTA_ARTEFACTOS / "historial_entrenamiento.csv")
        print("Exportados: modelo_fraude.pth, config_inferencia.json")


if __name__ == "__main__":
    principal()   # python -m nucleo.entrenamiento
