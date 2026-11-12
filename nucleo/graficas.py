"""Gráficas y tabla comparativa del experimento (Capa 1B – Persona 2).

Lee artefactos/historial_<etiqueta>.csv y artefactos/resultado_<etiqueta>.json
(los escribe nucleo/entrenamiento.py) y genera en artefactos/figuras/:
  loss_<etiqueta>.png          loss train vs validación (una por experimento)
  pr_auc_por_epoca.png         PR-AUC de validación por época, todos los experimentos
  comparativa_perdidas.png     precisión / recall / F1 / PR-AUC por experimento
y artefactos/tabla_comparativa.csv.

Uso:  python -m nucleo.graficas
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin ventana: solo guarda PNG
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
ARTEFACTOS = RAIZ / "artefactos"


def _historiales(carpeta: Path) -> dict:
    out = {}
    for ruta in sorted(carpeta.glob("historial_*.csv")):
        if ruta.name == "historial_entrenamiento.csv":   # copia del exportado, se omite
            continue
        out[ruta.stem.replace("historial_", "")] = pd.read_csv(ruta)
    return out


def curvas_loss(carpeta: Path = ARTEFACTOS) -> list:
    figuras = carpeta / "figuras"
    figuras.mkdir(exist_ok=True)
    rutas = []
    for etiqueta, h in _historiales(carpeta).items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["epoca"], h["loss_train"], marker="o", ms=3, label="Entrenamiento")
        ax.plot(h["epoca"], h["loss_val"], marker="s", ms=3, label="Validación")
        mejor = int(h["pr_auc_val"].idxmax())
        ax.axvline(h["epoca"][mejor], ls="--", color="gray", lw=1, label="Mejor PR-AUC (modelo guardado)")
        ax.set_xlabel("Época"); ax.set_ylabel("Pérdida")
        ax.set_title(f"Pérdida entrenamiento vs validación\n{etiqueta}", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        ruta = figuras / f"loss_{etiqueta}.png"
        fig.savefig(ruta, dpi=200); plt.close(fig)
        rutas.append(ruta)
    return rutas


def pr_auc_por_epoca(carpeta: Path = ARTEFACTOS) -> Path | None:
    hs = _historiales(carpeta)
    if not hs:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for etiqueta, h in hs.items():
        ax.plot(h["epoca"], h["pr_auc_val"], label=etiqueta)
    ax.set_xlabel("Época"); ax.set_ylabel("PR-AUC (validación)")
    ax.set_title("PR-AUC de validación por época", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout()
    ruta = carpeta / "figuras" / "pr_auc_por_epoca.png"
    (carpeta / "figuras").mkdir(exist_ok=True)
    fig.savefig(ruta, dpi=200); plt.close(fig)
    return ruta


def tabla_y_comparativa(carpeta: Path = ARTEFACTOS):
    filas = [json.load(open(r, encoding="utf-8")) for r in sorted(carpeta.glob("resultado_*.json"))]
    if not filas:
        return None
    df = pd.DataFrame(filas)
    cols = ["etiqueta", "perdida", "semilla", "optimizador", "ocultas", "parametros", "umbral",
            "precision", "recall", "f1", "pr_auc", "epoca_mejor", "epocas_corridas"]
    df = df[[c for c in cols if c in df.columns]].sort_values("pr_auc", ascending=False)
    df.to_csv(carpeta / "tabla_comparativa.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(df)), 4.5))
    ancho, x = 0.2, range(len(df))
    for i, (col, nombre) in enumerate([("precision", "Precisión"), ("recall", "Recall"),
                                       ("f1", "F1"), ("pr_auc", "PR-AUC")]):
        ax.bar([j + i * ancho for j in x], df[col], ancho, label=nombre)
    ax.set_xticks([j + 1.5 * ancho for j in x])
    ax.set_xticklabels(df["etiqueta"], rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("Valor (clase fraude, validación, umbral propio)")
    ax.set_title("Comparación de funciones de costo", fontsize=10)
    ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    (carpeta / "figuras").mkdir(exist_ok=True)
    fig.savefig(carpeta / "figuras" / "comparativa_perdidas.png", dpi=200); plt.close(fig)
    return df


def tabla_semillas(carpeta: Path = ARTEFACTOS) -> pd.DataFrame | None:
    """Resume media y desviación muestral de las corridas identificadas por semilla."""
    filas = [json.load(open(r, encoding="utf-8")) for r in sorted(carpeta.glob("resultado_*.json"))]
    df = pd.DataFrame(filas)
    if "semilla" not in df or df["semilla"].isna().all():
        return None
    df = df.dropna(subset=["semilla"])
    resumen = (df.groupby("perdida")[["f1", "pr_auc"]]
                 .agg(["mean", "std", "count"])
                 .reset_index())
    resumen.columns = ["perdida", "f1_media", "f1_desviacion", "f1_n",
                       "pr_auc_media", "pr_auc_desviacion", "pr_auc_n"]
    resumen.to_csv(carpeta / "tabla_semillas.csv", index=False)
    return resumen


if __name__ == "__main__":
    print("Curvas de loss:", [r.name for r in curvas_loss()])
    print("PR-AUC por época:", pr_auc_por_epoca())
    t = tabla_y_comparativa()
    print(t.to_string(index=False) if t is not None else "Aún no hay resultado_*.json: corre los experimentos.")
    s = tabla_semillas()
    print("Resumen por semillas:")
    print(s.to_string(index=False) if s is not None else "Aún no hay resultados con semilla.")
