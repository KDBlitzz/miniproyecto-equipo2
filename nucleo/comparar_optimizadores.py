"""Dinámica de optimización: SGD con momentum vs AdamW (Capa 1A – Persona 1).

Compara dos corridas idénticas en todo (Focal Loss alpha 0.75, 29→64→32→1, semilla 42,
lote 512, weight decay 1e-4) salvo el optimizador:
    AdamW  lr = 0.001
    SGD    lr = 0.01, momentum 0.9
Lee los historiales que escribe nucleo/entrenamiento.py y genera
artefactos/figuras/06_sgd_vs_adamw.png y artefactos/tabla_optimizadores.csv.

Uso:  python -m nucleo.comparar_optimizadores
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from nucleo.datos import RUTA_ARTEFACTOS

CORRIDAS = {"AdamW (lr 0.001)": "focal_a0.75_adamw_64-32_s42",
            "SGD + momentum (lr 0.01)": "focal_a0.75_sgd_64-32_s42"}
COLORES = {"AdamW (lr 0.001)": "#C44E52", "SGD + momentum (lr 0.01)": "#4C72B0"}


def principal() -> None:
    historiales = {n: pd.read_csv(RUTA_ARTEFACTOS / f"historial_{e}.csv") for n, e in CORRIDAS.items()}
    resultados = {n: json.loads((RUTA_ARTEFACTOS / f"resultado_{e}.json").read_text(encoding="utf-8"))
                  for n, e in CORRIDAS.items()}

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))
    for nombre, h in historiales.items():
        c = COLORES[nombre]
        ejes[0].plot(h["epoca"], h["loss_train"], color=c, label=nombre)
        ejes[1].plot(h["epoca"], h["loss_val"], color=c, label=nombre)
        ejes[2].plot(h["epoca"], h["pr_auc_val"], color=c, label=nombre)
        mejor = int(h["pr_auc_val"].idxmax())
        ejes[2].plot(h["epoca"][mejor], h["pr_auc_val"][mejor], "o", color=c)
    ejes[0].set_title("Pérdida de entrenamiento"); ejes[0].set_yscale("log")
    ejes[1].set_title("Pérdida de validación"); ejes[1].set_yscale("log")
    ejes[2].set_title("PR-AUC de validación (● = mejor época)")
    for eje in ejes:
        eje.set_xlabel("Época"); eje.grid(alpha=0.3); eje.legend(fontsize=8)
    fig.suptitle("SGD vs AdamW – Focal Loss, misma arquitectura y semilla", fontsize=11)
    fig.tight_layout()
    ruta = RUTA_ARTEFACTOS / "figuras" / "06_sgd_vs_adamw.png"
    fig.savefig(ruta, dpi=150); plt.close(fig)

    filas = []
    for nombre, h in historiales.items():
        r = resultados[nombre]
        mejor = int(h["pr_auc_val"].idxmax())
        # Primera época en que cada optimizador alcanza un PR-AUC de 0.64 (velocidad de convergencia)
        alcanza = h.index[h["pr_auc_val"] >= 0.64]
        filas.append({"optimizador": nombre, "epocas_corridas": len(h), "mejor_epoca": int(h["epoca"][mejor]),
                      "pr_auc_val": round(r["pr_auc"], 4), "f1_val": round(r["f1"], 4),
                      "precision_val": round(r["precision"], 4), "recall_val": round(r["recall"], 4),
                      "loss_train_final": round(float(h["loss_train"].iloc[-1]), 6),
                      "loss_val_final": round(float(h["loss_val"].iloc[-1]), 6),
                      "loss_val_minima": round(float(h["loss_val"].min()), 6),
                      "epoca_loss_val_minima": int(h["epoca"][int(h["loss_val"].idxmin())]),
                      "epoca_pr_auc_0.64": int(h["epoca"][alcanza[0]]) if len(alcanza) else None})
    tabla = pd.DataFrame(filas).set_index("optimizador")
    tabla.to_csv(RUTA_ARTEFACTOS / "tabla_optimizadores.csv")
    print(tabla.T.to_string())
    print("\nFigura:", ruta)


if __name__ == "__main__":
    principal()
