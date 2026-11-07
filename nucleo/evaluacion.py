"""Evaluación para clases desbalanceadas (Capa 1A – Persona 1).

Con 0.17 % de fraude, la accuracy no sirve: un modelo que responda "legítima" a todo
acierta el 99.83 % y no detecta ningún fraude. Aquí se reportan las métricas que sí
miden lo que importa en una pasarela de pagos:

- Recall (sensibilidad) de fraude : de los fraudes reales, cuántos se bloquearon.
- Precision de fraude             : de lo bloqueado, cuánto era fraude de verdad
                                    (lo demás son clientes legítimos molestos).
- F1                              : media armónica de las dos anteriores.
- PR-AUC (average precision)      : área bajo la curva precision-recall; es la métrica
                                    de resumen recomendada para datos desbalanceados
                                    (Saito y Rehmsmeier, 2015).
- ROC-AUC                         : se reporta por costumbre, pero con desbalance extremo
                                    sale optimista.
- Matriz de confusión             : TN, FP, FN, TP para un umbral dado.

El umbral de decisión no tiene por qué ser 0.5: se elige sobre VALIDACIÓN (nunca sobre test)
maximizando F1 o minimizando un costo de negocio, y se guarda en config_inferencia.json
para que la API lo use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def _a_numpy(x) -> np.ndarray:
    """Acepta tensores de PyTorch o arreglos y devuelve un vector NumPy 1-D."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def calcular_metricas(y_real, probabilidades, umbral: float = 0.5) -> dict:
    """Devuelve todas las métricas para un umbral dado.

    y_real: etiquetas 0/1; probabilidades: salida de la sigmoide en [0, 1].
    """
    y = _a_numpy(y_real).astype(int)
    p = _a_numpy(probabilidades)
    y_pred = (p >= umbral).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "umbral": float(umbral),
        "accuracy": float((tp + tn) / len(y)),  # se reporta solo para mostrar que engaña
        "precision_fraude": float(precision),
        "recall_fraude": float(recall),
        "f1_fraude": float(f1),
        "especificidad": float(especificidad),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if y.min() != y.max() else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def barrido_umbral(y_real, probabilidades, umbrales=None) -> pd.DataFrame:
    """Evalúa precision / recall / F1 / FP / FN para una rejilla de umbrales."""
    if umbrales is None:
        umbrales = np.round(np.arange(0.05, 0.96, 0.05), 2)
    filas = []
    for u in umbrales:
        m = calcular_metricas(y_real, probabilidades, u)
        filas.append({k: m[k] for k in ["umbral", "precision_fraude", "recall_fraude",
                                         "f1_fraude", "fp", "fn", "tp"]})
    return pd.DataFrame(filas).set_index("umbral")


def elegir_umbral(y_real, probabilidades, criterio: str = "f1",
                  costo_fn: float = 10.0, costo_fp: float = 1.0) -> tuple[float, dict]:
    """Elige el umbral sobre el conjunto de VALIDACIÓN.

    criterio = "f1"    : maximiza F1 de la clase fraude (equilibrio entre bloquear fraudes
                         y no molestar clientes legítimos).
    criterio = "costo" : minimiza costo_fn * FN + costo_fp * FP. Por defecto dejar pasar un
                         fraude cuesta 10 veces más que bloquear un pago legítimo.
    Se recorre una rejilla fina (0.01) sobre los puntos de la curva precision-recall.
    Devuelve (umbral, métricas en ese umbral).
    """
    y = _a_numpy(y_real).astype(int)
    p = _a_numpy(probabilidades)
    candidatos = np.round(np.arange(0.01, 1.00, 0.01), 2)
    mejor_u, mejor_valor, mejor_m = 0.5, None, None
    for u in candidatos:
        m = calcular_metricas(y, p, u)
        if criterio == "f1":
            valor = m["f1_fraude"]
            mejor = mejor_valor is None or valor > mejor_valor
        elif criterio == "costo":
            valor = costo_fn * m["fn"] + costo_fp * m["fp"]
            mejor = mejor_valor is None or valor < mejor_valor
        else:
            raise ValueError("criterio debe ser 'f1' o 'costo'")
        if mejor:
            mejor_u, mejor_valor, mejor_m = float(u), valor, m
    return mejor_u, mejor_m


# ---------------------------------------------------------------- figuras para el informe

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def graficar_matriz_confusion(y_real, probabilidades, umbral: float, ruta_salida: Path,
                              titulo: str = "Matriz de confusión (test)") -> Path:
    plt = _plt()
    m = calcular_metricas(y_real, probabilidades, umbral)
    matriz = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(matriz, cmap="Blues")
    etiquetas = ["Legítima", "Fraude"]
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Pred. {e}" for e in etiquetas])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"Real {e}" for e in etiquetas])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matriz[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if matriz[i, j] > matriz.max() / 2 else "black")
    ax.set_title(f"{titulo}\numbral = {umbral:.2f} | recall = {m['recall_fraude']:.3f} | "
                 f"precision = {m['precision_fraude']:.3f}", fontsize=10)
    fig.tight_layout()
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150); plt.close(fig)
    return Path(ruta_salida)


def graficar_curva_pr(y_real, probabilidades, ruta_salida: Path, umbral: float | None = None,
                      etiqueta: str = "modelo") -> Path:
    """Curva precision-recall con la línea base (prevalencia) y el punto del umbral elegido."""
    plt = _plt()
    y = _a_numpy(y_real).astype(int); p = _a_numpy(probabilidades)
    precision, recall, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, color="#C44E52", label=f"{etiqueta} (PR-AUC = {ap:.3f})")
    ax.axhline(y.mean(), color="gray", ls="--", label=f"azar (prevalencia = {y.mean():.4f})")
    if umbral is not None:
        m = calcular_metricas(y, p, umbral)
        ax.plot(m["recall_fraude"], m["precision_fraude"], "o", color="black",
                label=f"umbral {umbral:.2f} (P={m['precision_fraude']:.2f}, R={m['recall_fraude']:.2f})")
    ax.set_xlabel("Recall (fraudes detectados)"); ax.set_ylabel("Precision (bloqueos acertados)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_title("Curva precision-recall"); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150); plt.close(fig)
    return Path(ruta_salida)


def graficar_barrido_umbral(y_real, probabilidades, ruta_salida: Path,
                            umbral_elegido: float | None = None) -> Path:
    """Precision, recall y F1 en función del umbral: muestra el compromiso que se decide."""
    plt = _plt()
    tabla = barrido_umbral(y_real, probabilidades, np.round(np.arange(0.01, 1.0, 0.01), 2))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(tabla.index, tabla["precision_fraude"], label="Precision")
    ax.plot(tabla.index, tabla["recall_fraude"], label="Recall")
    ax.plot(tabla.index, tabla["f1_fraude"], label="F1", lw=2)
    if umbral_elegido is not None:
        ax.axvline(umbral_elegido, color="black", ls="--", label=f"umbral elegido = {umbral_elegido:.2f}")
    ax.set_xlabel("Umbral de decisión"); ax.set_ylabel("Valor"); ax.set_ylim(0, 1.05)
    ax.set_title("Efecto del umbral sobre la clase fraude (validación)"); ax.legend(fontsize=9)
    fig.tight_layout()
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta_salida, dpi=150); plt.close(fig)
    return Path(ruta_salida)


def predecir_probabilidades(modelo, loader) -> tuple[np.ndarray, np.ndarray]:
    """Pasa un DataLoader completo por el modelo en modo evaluación y sin gradiente.

    Devuelve (probabilidades, etiquetas). Lo usan el Entrenador (P2), el notebook y
    los experimentos de P1; es el "módulo de validación" del enunciado.
    """
    import torch
    modelo.eval()                      # desactiva Dropout y fija BatchNorm
    probabilidades, etiquetas = [], []
    with torch.no_grad():              # no se construye el grafo: más rápido y sin fugas de memoria
        for X, y in loader:
            logits = modelo(X)
            probabilidades.append(torch.sigmoid(logits))
            etiquetas.append(y)
    return (_a_numpy(torch.cat(probabilidades)), _a_numpy(torch.cat(etiquetas)))


if __name__ == "__main__":
    # Prueba de humo: evalúa el modelo SIN entrenar sobre validación. Las métricas deben salir
    # al nivel del azar (PR-AUC ≈ 0.0017 = prevalencia); sirve para verificar que el módulo
    # funciona antes de que exista un modelo entrenado.
    import torch
    from nucleo.datos import crear_dataloaders, RUTA_ARTEFACTOS
    from nucleo.modelo import DetectorFraudeMLP

    torch.manual_seed(42)
    datos = crear_dataloaders()
    modelo = DetectorFraudeMLP()
    prob_val, y_val = predecir_probabilidades(modelo, datos["loader_val"])
    print(f"Probabilidades de validación: {prob_val.shape}, rango [{prob_val.min():.3f}, {prob_val.max():.3f}]")

    m = calcular_metricas(y_val, prob_val, 0.5)
    print("\nMétricas con umbral 0.5 (modelo sin entrenar, referencia de azar):")
    for k in ["accuracy", "precision_fraude", "recall_fraude", "f1_fraude", "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"]:
        print(f"  {k:18s} {m[k]:.4f}" if isinstance(m[k], float) else f"  {k:18s} {m[k]}")

    u, mu = elegir_umbral(y_val, prob_val, "f1")
    print(f"\nUmbral que maximiza F1 en validación: {u:.2f} → F1 = {mu['f1_fraude']:.4f}")
    print("\nBarrido de umbral (resumen):")
    print(barrido_umbral(y_val, prob_val, [0.1, 0.3, 0.5, 0.7, 0.9]).round(4))

    carpeta = RUTA_ARTEFACTOS / "figuras"
    print("\nFiguras de prueba:")
    print(" ", graficar_matriz_confusion(y_val, prob_val, u, carpeta / "prueba_matriz_confusion.png", "Matriz de confusión (val, sin entrenar)"))
    print(" ", graficar_curva_pr(y_val, prob_val, carpeta / "prueba_curva_pr.png", u, "sin entrenar"))
    print(" ", graficar_barrido_umbral(y_val, prob_val, carpeta / "prueba_barrido_umbral.png", u))
