"""Ingesta y preprocesamiento (Capa 1A – Persona 1).

Responsabilidades:
1. Descargar el dataset Credit Card Fraud (ULB) y guardarlo en data/creditcard.csv.
2. Análisis exploratorio del desbalance (conteos, porcentajes, distribución de Amount).
3. Partición estratificada train / val / test (70 / 15 / 15) con semilla 42.
4. Escalador Z-score ajustado SOLO con train y guardado en artefactos/escalador.pkl.
5. TransaccionesDataset(torch.utils.data.Dataset) y DataLoaders por mini-batches.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Rutas relativas a la raíz del repositorio
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "data" / "creditcard.csv"
RUTA_ARTEFACTOS = RAIZ / "artefactos"
SEMILLA = 42

# Columnas del dataset (versión OpenML): V1..V28 y Amount como entradas (29) y Class como
# etiqueta (1 = fraude). La versión de OpenML no incluye "Time"; esa columna solo indica los
# segundos transcurridos desde la primera transacción del registro y no aporta información
# útil para evaluar una transacción aislada en una pasarela de pagos.
COLUMNAS_ENTRADA = [f"V{i}" for i in range(1, 29)] + ["Amount"]
N_CARACTERISTICAS = len(COLUMNAS_ENTRADA)  # 29
COLUMNA_ETIQUETA = "Class"


def descargar_dataset(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Descarga el dataset desde OpenML (sin cuenta de Kaggle) y lo guarda como CSV.

    Si el CSV ya existe, solo lo lee. Devuelve un DataFrame con 30 columnas:
    V1..V28, Amount y Class (0 = legítima, 1 = fraude).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        df = pd.read_csv(ruta)
    else:
        from sklearn.datasets import fetch_openml
        print("Descargando Credit Card Fraud (ULB) desde OpenML (~150 MB)...")
        datos = fetch_openml("creditcard", version=1, as_frame=True, parser="auto")
        df = datos.frame.copy()
        # OpenML entrega Class como categoría de texto ("0"/"1"); se convierte a entero
        df[COLUMNA_ETIQUETA] = df[COLUMNA_ETIQUETA].astype(str).astype(int)
        df.to_csv(ruta, index=False)
        print(f"Guardado en {ruta}")
    # Verificación de integridad: columnas esperadas y sin valores faltantes
    faltantes = [c for c in COLUMNAS_ENTRADA + [COLUMNA_ETIQUETA] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {faltantes}")
    if df.isna().any().any():
        raise ValueError("El dataset contiene valores faltantes")
    return df


def resumen_desbalance(df: pd.DataFrame) -> dict:
    """Devuelve conteo y porcentaje por clase, razón negativos/positivos y estadísticas de Amount."""
    conteo = df[COLUMNA_ETIQUETA].value_counts().sort_index()
    n_total = int(len(df))
    n_legitimas = int(conteo.get(0, 0))
    n_fraudes = int(conteo.get(1, 0))
    return {
        "total": n_total,
        "legitimas": n_legitimas,
        "fraudes": n_fraudes,
        "porcentaje_fraude": 100.0 * n_fraudes / n_total,
        # Este cociente es el pos_weight que usará la BCE ponderada de la Persona 2
        "razon_neg_pos": n_legitimas / n_fraudes,
        "monto_por_clase": df.groupby(COLUMNA_ETIQUETA)["Amount"].describe(),
    }


def graficar_desbalance(df: pd.DataFrame, carpeta: Path = RUTA_ARTEFACTOS / "figuras") -> list[Path]:
    """Genera las figuras del análisis exploratorio para el informe. Devuelve las rutas."""
    import matplotlib
    matplotlib.use("Agg")  # sin ventana: solo guarda archivos
    import matplotlib.pyplot as plt

    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = []

    # Figura 1: conteo por clase en escala logarítmica (en escala lineal el fraude no se ve)
    conteo = df[COLUMNA_ETIQUETA].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(["Legítima (0)", "Fraude (1)"], conteo.values, color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Transacciones (escala log)")
    ax.set_title("Desbalance de clases en el dataset")
    for barra, valor in zip(barras, conteo.values):
        ax.text(barra.get_x() + barra.get_width() / 2, valor, f"{valor:,}\n({100*valor/len(df):.3f} %)",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    ruta = carpeta / "01_desbalance_clases.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    rutas.append(ruta)

    # Figura 2: distribución del monto por clase (log1p porque Amount tiene cola larga)
    fig, ax = plt.subplots(figsize=(7, 4))
    for clase, color, etiqueta in [(0, "#4C72B0", "Legítima"), (1, "#C44E52", "Fraude")]:
        montos = np.log1p(df.loc[df[COLUMNA_ETIQUETA] == clase, "Amount"])
        ax.hist(montos, bins=50, density=True, alpha=0.6, color=color, label=etiqueta)
    ax.set_xlabel("log(1 + Amount)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del monto por clase")
    ax.legend()
    fig.tight_layout()
    ruta = carpeta / "02_monto_por_clase.png"
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    rutas.append(ruta)
    return rutas


def particionar(df: pd.DataFrame, semilla: int = SEMILLA):
    """Partición estratificada 70 / 15 / 15 (train / val / test).

    Estratificar significa que los tres subconjuntos conservan la misma proporción de
    fraude (0.173 %). Sin esto, con 492 positivos, una partición aleatoria podría dejar
    el conjunto de validación casi sin fraudes. Devuelve seis arreglos NumPy float32:
    (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    from sklearn.model_selection import train_test_split

    X = df[COLUMNAS_ENTRADA].to_numpy(dtype=np.float32)
    y = df[COLUMNA_ETIQUETA].to_numpy(dtype=np.float32)
    # Primer corte: 70 % train, 30 % temporal
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=semilla)
    # Segundo corte: el 30 % temporal se divide a la mitad → 15 % val, 15 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=semilla)
    return X_train, y_train, X_val, y_val, X_test, y_test


def resumen_particion(y_train, y_val, y_test) -> pd.DataFrame:
    """Tabla con tamaño, fraudes y porcentaje de fraude por subconjunto (para el informe)."""
    filas = []
    for nombre, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        n, f = len(y), int(y.sum())
        filas.append({"conjunto": nombre, "transacciones": n, "fraudes": f,
                      "porcentaje_fraude": round(100.0 * f / n, 3)})
    return pd.DataFrame(filas).set_index("conjunto")


class Escalador:
    """Estandarización Z-score: z = (x - media) / desviación.

    Se ajusta únicamente con el conjunto de entrenamiento (si se usara todo el dataset,
    información de validación y prueba se "filtraría" al entrenamiento). Se guarda en
    artefactos/escalador.pkl para que la API aplique exactamente la misma transformación
    a cada transacción nueva: el requisito de "guardar los escaladores empíricos".
    """

    def __init__(self) -> None:
        self.media: np.ndarray | None = None
        self.desviacion: np.ndarray | None = None
        self.columnas: list[str] = list(COLUMNAS_ENTRADA)

    def ajustar(self, X: np.ndarray) -> "Escalador":
        self.media = X.mean(axis=0).astype(np.float32)
        self.desviacion = X.std(axis=0).astype(np.float32)
        # Una columna constante tendría desviación 0 → división por cero; se protege con 1
        self.desviacion[self.desviacion == 0] = 1.0
        return self

    def transformar(self, X: np.ndarray) -> np.ndarray:
        if self.media is None:
            raise RuntimeError("El escalador no ha sido ajustado: llame a ajustar(X_train) primero")
        return ((np.asarray(X, dtype=np.float32) - self.media) / self.desviacion).astype(np.float32)

    def ajustar_transformar(self, X: np.ndarray) -> np.ndarray:
        return self.ajustar(X).transformar(X)

    def guardar(self, ruta: Path = RUTA_ARTEFACTOS / "escalador.pkl") -> Path:
        import joblib
        ruta.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"media": self.media, "desviacion": self.desviacion, "columnas": self.columnas}, ruta)
        return ruta

    @classmethod
    def cargar(cls, ruta: Path = RUTA_ARTEFACTOS / "escalador.pkl") -> "Escalador":
        import joblib
        datos = joblib.load(ruta)
        escalador = cls()
        escalador.media = datos["media"]
        escalador.desviacion = datos["desviacion"]
        escalador.columnas = datos["columnas"]
        return escalador


class TransaccionesDataset(Dataset):
    """Dataset personalizado (hereda de torch.utils.data.Dataset).

    Recibe las características YA escaladas y las etiquetas, y entrega por índice
    un par (tensor de 29 características, etiqueta float con forma (1,)).
    La etiqueta se entrega con forma (1,) para que coincida con el logit de salida
    del modelo, de forma (batch, 1), sin reshapes dentro del bucle de entrenamiento.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError(f"X tiene {len(X)} filas y y tiene {len(y)}")
        # Se convierten una sola vez a tensores (evita convertir en cada __getitem__)
        self.X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        self.y = torch.as_tensor(np.asarray(y, dtype=np.float32)).unsqueeze(1)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, indice: int):
        return self.X[indice], self.y[indice]

    @property
    def n_positivos(self) -> int:
        return int(self.y.sum().item())

    @property
    def n_negativos(self) -> int:
        return len(self) - self.n_positivos


def crear_dataloaders(tamano_lote: int = 512, semilla: int = SEMILLA, df: pd.DataFrame | None = None):
    """Pipeline completo: descarga → partición → escalador → Datasets → DataLoaders.

    Devuelve un diccionario con:
      loader_train, loader_val, loader_test : DataLoaders (shuffle solo en train)
      dataset_train, dataset_val, dataset_test
      escalador                             : Escalador ajustado con train (ya guardado en .pkl)
      X_test_original                       : test SIN escalar, para que la API/dashboard lo
                                              usen como transacciones "crudas" entrantes
      y_test
    """
    torch.manual_seed(semilla)
    if df is None:
        df = descargar_dataset()
    X_train, y_train, X_val, y_val, X_test, y_test = particionar(df, semilla)

    escalador = Escalador().ajustar(X_train)
    escalador.guardar()

    dataset_train = TransaccionesDataset(escalador.transformar(X_train), y_train)
    dataset_val = TransaccionesDataset(escalador.transformar(X_val), y_val)
    dataset_test = TransaccionesDataset(escalador.transformar(X_test), y_test)

    # Generador con semilla para que el barajado de train sea reproducible
    generador = torch.Generator().manual_seed(semilla)
    loader_train = DataLoader(dataset_train, batch_size=tamano_lote, shuffle=True, generator=generador)
    loader_val = DataLoader(dataset_val, batch_size=tamano_lote * 4, shuffle=False)
    loader_test = DataLoader(dataset_test, batch_size=tamano_lote * 4, shuffle=False)

    return {
        "loader_train": loader_train, "loader_val": loader_val, "loader_test": loader_test,
        "dataset_train": dataset_train, "dataset_val": dataset_val, "dataset_test": dataset_test,
        "escalador": escalador, "X_test_original": X_test, "y_test": y_test,
    }


def exportar_muestra_test(X_test: np.ndarray, y_test: np.ndarray, n_legitimas: int = 1000,
                          semilla: int = SEMILLA, ruta: Path = RUTA_ARTEFACTOS / "muestra_test.csv") -> Path:
    """Guarda una muestra del test set SIN escalar (todos los fraudes + n legítimas).

    La usan la Capa 2 (simular_rafaga.py) y la Capa 3 (simulador del dashboard) como
    transacciones entrantes reales. Al no estar escaladas, obligan a la API a aplicar
    escalador.pkl, que es exactamente lo que se quiere probar.
    """
    rng = np.random.default_rng(semilla)
    idx_fraude = np.flatnonzero(y_test == 1)
    idx_legit = rng.choice(np.flatnonzero(y_test == 0), size=n_legitimas, replace=False)
    idx = rng.permutation(np.concatenate([idx_fraude, idx_legit]))
    muestra = pd.DataFrame(X_test[idx], columns=COLUMNAS_ENTRADA)
    muestra.insert(0, "id_transaccion", [f"TX-{i:06d}" for i in range(len(idx))])
    muestra[COLUMNA_ETIQUETA] = y_test[idx].astype(int)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(ruta, index=False)
    return ruta



if __name__ == "__main__":
    # Paso 1: python -m nucleo.datos  → descarga, resumen del desbalance y figuras
    df = descargar_dataset()
    print("Forma del dataset:", df.shape)
    resumen = resumen_desbalance(df)
    print(f"Total: {resumen['total']:,} | Legítimas: {resumen['legitimas']:,} | "
          f"Fraudes: {resumen['fraudes']:,} ({resumen['porcentaje_fraude']:.3f} %)")
    print(f"Razón negativos/positivos (pos_weight): {resumen['razon_neg_pos']:.1f}")
    print("\nAmount por clase:")
    print(resumen["monto_por_clase"].round(2))
    for ruta in graficar_desbalance(df):
        print("Figura guardada:", ruta)

    # Paso 2: partición estratificada 70 / 15 / 15
    X_train, y_train, X_val, y_val, X_test, y_test = particionar(df)
    print("\nPartición estratificada:")
    print(resumen_particion(y_train, y_val, y_test))

    # Paso 3: escalador Z-score ajustado solo con train y guardado en artefactos/
    escalador = Escalador().ajustar(X_train)
    ruta_esc = escalador.guardar()
    Z_train = escalador.transformar(X_train)
    Z_val = escalador.transformar(X_val)
    print(f"\nEscalador guardado en {ruta_esc}")
    print(f"Train escalado → media promedio {Z_train.mean():+.4f}, desviación promedio {Z_train.std(axis=0).mean():.4f}")
    print(f"Val   escalado → media promedio {Z_val.mean():+.4f}, desviación promedio {Z_val.std(axis=0).mean():.4f}")
    # Comprobación de que el .pkl reproduce exactamente la misma transformación
    recargado = Escalador.cargar(ruta_esc)
    print("El escalador recargado reproduce la transformación:",
          bool(np.allclose(recargado.transformar(X_val), Z_val)))

    # Paso 4: Datasets y DataLoaders por mini-batches
    datos = crear_dataloaders(tamano_lote=512, df=df)
    lote_X, lote_y = next(iter(datos["loader_train"]))
    print(f"\nDataLoaders: train {len(datos['loader_train'])} lotes, val {len(datos['loader_val'])}, "
          f"test {len(datos['loader_test'])} (batch train = 512)")
    print(f"Primer lote: X {tuple(lote_X.shape)} {lote_X.dtype}, y {tuple(lote_y.shape)} {lote_y.dtype}")
    # Cuántos lotes de una época NO contienen ningún fraude (dato clave para P2 y la reflexión)
    lotes_sin_fraude = sum(int(y.sum().item() == 0) for _, y in datos["loader_train"])
    print(f"Lotes de train sin ningún fraude: {lotes_sin_fraude} de {len(datos['loader_train'])} "
          f"(esperado ≈ {512 * 344 / 199364:.2f} fraudes por lote)")
    ruta_muestra = exportar_muestra_test(datos["X_test_original"], datos["y_test"])
    print(f"Muestra de test para API/dashboard guardada en {ruta_muestra}")
