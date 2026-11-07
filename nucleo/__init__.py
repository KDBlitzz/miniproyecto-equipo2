"""Núcleo PyTorch del sistema de detección de fraude (Capa 1).

Módulos:
- datos         : descarga, EDA del desbalance, partición, escalador y Dataset/DataLoader  (Persona 1)
- modelo        : DetectorFraudeMLP(nn.Module)                                             (Persona 2)
- perdidas      : FocalLoss y BCE ponderada                                                (Persona 2)
- entrenamiento : Entrenador con bucle explícito y validación con no_grad                  (Persona 2)
- evaluacion    : métricas para clases desbalanceadas, umbral, matriz de confusión          (Persona 1)
"""
