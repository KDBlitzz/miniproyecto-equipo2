# Mini Proyecto – Equipo 2: Riesgo Crediticio y Detección de Fraude

Sistema inteligente de tres capas para una pasarela de pagos: una red neuronal en PyTorch
entrenada con una función de costo modificada para el desbalance de clases (Capa 1), una API
FastAPI que responde `approved` / `blocked` en milisegundos (Capa 2) y un dashboard bancario
ejecutivo en Streamlit con alertas en tiempo real (Capa 3).

**Curso:** Redes Neuronales y Aprendizaje Profundo – UNITEC – Sección 1905 – Ing. Kevin Fúnez
**Equipo 2:** Brandon Hernández, Jesús Rodríguez, Walter Doblado, Diego Lara, Carlos Flores

## Estructura

| Carpeta | Capa | Qué contiene | Responsable |
|---|---|---|---|
| `nucleo/` | 1 | Paquete Python compartido: datos, modelo, pérdidas, entrenamiento, evaluación | Persona 1 (datos/evaluación) y Persona 2 (modelo/pérdida/entrenamiento) |
| `notebooks/` | 1 | Notebook de Colab (estilo de clase) que importa `nucleo/` y muestra resultados | Persona 1 y 2 |
| `artefactos/` | 1→2 | `modelo_fraude.pth`, `escalador.pkl`, `config_inferencia.json`, historial, figuras | Persona 2 (modelo) / Persona 1 (escalador, métricas) |
| `api/` | 2 | FastAPI: pasarela de pagos | Persona 3 |
| `dashboard/` | 3 | Streamlit: dashboard bancario ejecutivo | Persona 4 |
| `informe/` | — | Informe Word | Persona 5 |
| `presentacion/` | — | Láminas (PDF exportado de Canva) | Persona 5 |
| `data/` | 1 | Dataset descargado (ignorado por git) | cada quien lo descarga |

## Instalación (una sola vez)

```bash
git clone https://github.com/<usuario>/miniproyecto-equipo2.git
cd miniproyecto-equipo2
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / Mac:
source venv/bin/activate
pip install -r requirements.txt
```

Descargar el dataset (284 807 transacciones, 492 fraudes) una sola vez:

```bash
python -c "from nucleo.datos import descargar_dataset; descargar_dataset()"
```

## Cómo correr las tres capas

```bash
# Capa 1 – entrenar y exportar artefactos (o abrir notebooks/ en VS Code / Colab)
python -m nucleo.entrenamiento

# Capa 2 – API en http://localhost:8000  (docs en /docs)
uvicorn api.main:app --reload

# Capa 3 – dashboard en http://localhost:8501
streamlit run dashboard/app.py
```

## Flujo de trabajo en git

- `main` siempre corre. Cada persona trabaja en su rama: `capa1-datos`, `capa1-modelo`, `capa2-api`, `capa3-dashboard`, `informe`.
- Commits pequeños y frecuentes; al terminar algo que funciona, *merge* a `main` (o Pull Request si prefieren revisión).
- `nucleo/` lo editan solo Persona 1 y Persona 2, coordinados; las demás capas lo importan.
- Las figuras y artefactos finales **sí** se commitean (son pequeños); el dataset **no**.

## Contrato de la API (no se modifica sin avisar al grupo)

`POST /api/v1/transacciones` → `{"id_transaccion", "monto", "tiempo", "caracteristicas": [28 valores V1..V28]}`
→ `{"id_transaccion", "status": "approved"|"blocked", "probabilidad_fraude", "umbral", "latencia_ms"}`

`POST /api/v1/transacciones/lote` → `{"transacciones": [...]}` → `{"resultados": [...], "latencia_total_ms", "procesadas"}`

`GET /api/v1/salud` → `{"status": "ok", "modelo", "umbral", "version"}`

Los detalles completos del reparto, decisiones y calendario están en `MiniProyecto_Equipo2_Reparto_de_partes.pdf`.
