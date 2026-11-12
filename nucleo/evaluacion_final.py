"""Evaluación final sobre el conjunto de PRUEBA (Capa 1A – Persona 1).

El umbral ya fue elegido sobre validación y vive en config_inferencia.json; aquí NO se
vuelve a ajustar nada: el test se usa una sola vez, para reportar el rendimiento real
del modelo exportado (modelo_fraude.pth), el mismo que sirve la API.

Uso:  python -m nucleo.evaluacion_final
Genera artefactos/metricas_test.json y tres figuras para el informe.
"""
import json

import torch

from nucleo.datos import RUTA_ARTEFACTOS, crear_dataloaders
from nucleo.evaluacion import (calcular_metricas, graficar_barrido_umbral, graficar_curva_pr,
                               graficar_matriz_confusion, predecir_probabilidades)
from nucleo.modelo import DetectorFraudeMLP

CLAVES = ["accuracy", "precision_fraude", "recall_fraude", "f1_fraude", "pr_auc", "roc_auc",
          "tn", "fp", "fn", "tp"]


def principal() -> None:
    torch.manual_seed(42)
    config = json.loads((RUTA_ARTEFACTOS / "config_inferencia.json").read_text(encoding="utf-8"))
    umbral = float(config["umbral"])

    # Se reconstruye la arquitectura desde el config y se cargan los pesos exportados
    modelo = DetectorFraudeMLP(config["dim_entrada"], tuple(config["dims_ocultas"]), config["p_dropout"])
    modelo.load_state_dict(torch.load(RUTA_ARTEFACTOS / "modelo_fraude.pth", map_location="cpu",
                                      weights_only=True))
    datos = crear_dataloaders()

    # Validación (donde se eligió el umbral) y prueba (datos que el modelo nunca vio)
    prob_val, y_val = predecir_probabilidades(modelo, datos["loader_val"])
    prob_test, y_test = predecir_probabilidades(modelo, datos["loader_test"])
    m_val = calcular_metricas(y_val, prob_val, umbral)
    m_test = calcular_metricas(y_test, prob_test, umbral)

    print(f"Modelo: pérdida {config.get('perdida')} | umbral {umbral:.2f} (elegido en validación)\n")
    print(f"{'métrica':18s} {'validación':>12s} {'prueba':>12s}")
    for k in CLAVES:
        fmt = "{:>12.4f}" if isinstance(m_test[k], float) else "{:>12d}"
        print(f"{k:18s} " + fmt.format(m_val[k]) + " " + fmt.format(m_test[k]))

    # Línea base trivial: responder "legítima" a todo
    trivial = 1.0 - float(y_test.mean())
    print(f"\nAccuracy de un modelo que aprueba todo: {trivial:.4f} (recall de fraude 0)")
    print(f"Fraudes en prueba: {int(y_test.sum())} | detectados: {m_test['tp']} | "
          f"escapados: {m_test['fn']} | legítimas bloqueadas: {m_test['fp']} de {m_test['tn'] + m_test['fp']}")

    salida = {"umbral": umbral, "perdida": config.get("perdida"),
              "validacion": m_val, "prueba": m_test, "accuracy_modelo_trivial": trivial}
    ruta = RUTA_ARTEFACTOS / "metricas_test.json"
    ruta.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado: {ruta}")

    figuras = RUTA_ARTEFACTOS / "figuras"
    print(graficar_matriz_confusion(y_test, prob_test, umbral, figuras / "03_matriz_confusion_test.png",
                                    "Matriz de confusión (prueba)"))
    print(graficar_curva_pr(y_test, prob_test, figuras / "04_curva_pr_test.png", umbral, "Focal Loss (prueba)"))
    print(graficar_barrido_umbral(y_val, prob_val, figuras / "05_barrido_umbral_validacion.png", umbral))


if __name__ == "__main__":
    principal()
