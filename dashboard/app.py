"""Dashboard Bancario Ejecutivo (Capa 3 – Persona 4).  Ejecutar: streamlit run dashboard/app.py

Simula un flujo de transacciones (desde artefactos/muestra_test.csv que entrega Persona 1),
las envía a la API y muestra KPIs, tabla en vivo, gráfica de probabilidad y una alerta
emergente inmediata cuando una transacción resulta `blocked`.

Fuentes de datos (se elige en la barra lateral):
  * API real  -> http://localhost:8000/api/v1 (Persona 3)
  * Mock local -> mismas respuestas del contrato 2.4, calculadas dentro del dashboard
"""
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mock_api  # noqa: E402  (lógica del mock; mismo contrato que la API real)

API = "http://localhost:8000/api/v1"
RAIZ = Path(__file__).resolve().parent.parent
RUTA_MUESTRA = RAIZ / "artefactos" / "muestra_test.csv"
COLS_V = [f"V{i}" for i in range(1, 29)]
MONEDA = "$"
MAX_HISTORIAL = 5000

# Campos cosméticos: NO entran al modelo, solo hacen que el dashboard "parezca banco".
COMERCIOS = ["Supermercado", "Farmacia", "Gasolinera", "Restaurante", "Comercio en línea",
             "Aerolínea", "Electrónica", "Cajero automático", "Hotel", "Ropa y calzado"]
CIUDADES = ["Tegucigalpa", "San Pedro Sula", "La Ceiba", "Choloma", "Comayagua",
            "Choluteca", "El Progreso", "Danlí"]

COLOR_OK, COLOR_BLOQ, COLOR_UMBRAL = "#2a7ab0", "#d62728", "#e69f00"

st.set_page_config(page_title="Dashboard Bancario – Equipo 2", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------ datos y estado
@st.cache_data
def cargar_muestra() -> pd.DataFrame:
    return pd.read_csv(RUTA_MUESTRA)


class ErrorAPI(Exception):
    """Cualquier fallo al hablar con la API (caída, 4xx, 5xx, respuesta rara)."""


def comprobar_salud() -> dict | None:
    try:
        r = httpx.get(f"{API}/salud", timeout=1.5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def iniciar_estado() -> None:
    if "historial" in st.session_state:
        return
    st.session_state.historial = []        # lista de dicts, uno por transacción procesada
    st.session_state.contador = 0
    st.session_state.corriendo = False
    st.session_state.alerta = None         # última transacción bloqueada (para el banner)
    st.session_state.error = None
    st.session_state.t0 = time.time()
    # Primera carga: API real si responde; si no, mock local (y el dashboard lo dice).
    st.session_state.fuente = "API real" if comprobar_salud() else "Mock local"


# ------------------------------------------------------------------ comunicación con la capa 2
def evaluar(txs: list[dict], lote: bool) -> list[dict]:
    """Envía transacciones (payload del contrato 2.4) y devuelve la lista de resultados."""
    if st.session_state.fuente == "Mock local":
        return mock_api.evaluar_muchas(txs)["resultados"]
    try:
        if lote:
            r = httpx.post(f"{API}/transacciones/lote", json={"transacciones": txs}, timeout=10)
            r.raise_for_status()
            return r.json()["resultados"]
        resultados = []
        for tx in txs:
            r = httpx.post(f"{API}/transacciones", json=tx, timeout=10)
            r.raise_for_status()
            resultados.append(r.json())
        return resultados
    except httpx.HTTPStatusError as e:
        raise ErrorAPI(f"La API respondió {e.response.status_code}. ¿Ya está implementado el endpoint? "
                       f"({e.response.text[:120]})") from e
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise ErrorAPI(f"No se pudo hablar con la API en {API}: {type(e).__name__}. "
                       "Levanta la API (o el mock) o cambia a «Mock local».") from e


def construir_transaccion(fila: pd.Series, forzar_id: str | None = None) -> tuple[dict, dict]:
    """De una fila del test set saca (payload para la API, metadatos cosméticos para la tabla)."""
    st.session_state.contador += 1
    id_tx = forzar_id or f"TX-{900000 + st.session_state.contador:06d}"
    payload = {
        "id_transaccion": id_tx,
        "monto": round(float(fila["Amount"]), 2),
        "tiempo": round(40000 + time.time() - st.session_state.t0, 1),
        "caracteristicas": [float(fila[c]) for c in COLS_V],
    }
    meta = {
        "Hora": datetime.now().strftime("%H:%M:%S"),
        "Comercio": random.choice(COMERCIOS),
        "Ciudad": random.choice(CIUDADES),
        "Real": "Fraude" if int(fila["Class"]) == 1 else "Legítima",
    }
    return payload, meta


def procesar(filas: list[pd.Series], lote: bool, inyectada: bool = False) -> None:
    """Manda las filas a la API, guarda resultados y dispara alertas para los blocked."""
    pares = [construir_transaccion(f) for f in filas]
    try:
        resultados = evaluar([p for p, _ in pares], lote)
    except ErrorAPI as e:
        st.session_state.error = str(e)
        st.session_state.corriendo = False  # pausa para no inundar de errores
        return
    st.session_state.error = None

    por_id = {r["id_transaccion"]: r for r in resultados}
    nuevos_bloqueos = []
    for payload, meta in pares:
        res = por_id.get(payload["id_transaccion"])
        if res is None:
            continue
        reg = {
            **meta,
            "ID": payload["id_transaccion"],
            "Monto": payload["monto"],
            "Probabilidad": float(res["probabilidad_fraude"]),
            "Umbral": float(res["umbral"]),
            "Status": res["status"],
            "Latencia (ms)": float(res["latencia_ms"]),
            "Origen": "Inyectada" if inyectada else "Flujo",
        }
        st.session_state.historial.append(reg)
        if reg["Status"] == "blocked":
            nuevos_bloqueos.append(reg)

    del st.session_state.historial[:-MAX_HISTORIAL]

    if nuevos_bloqueos:
        st.session_state.alerta = nuevos_bloqueos[-1]
        for reg in nuevos_bloqueos[:3]:  # en ráfagas grandes no llenamos la pantalla de toasts
            st.toast(f"**{reg['ID']}** · {MONEDA}{reg['Monto']:,.2f} · prob. {reg['Probabilidad']:.1%}",
                     icon="🚨")
        if len(nuevos_bloqueos) > 3:
            st.toast(f"…y {len(nuevos_bloqueos) - 3} bloqueos más en esta ráfaga", icon="🚨")
    elif inyectada:
        reg = st.session_state.historial[-1]
        st.toast(f"El modelo APROBÓ la transacción inyectada {reg['ID']} "
                 f"(prob. {reg['Probabilidad']:.1%}, umbral {reg['Umbral']:.2f}).", icon="⚠️")


def elegir_filas(df: pd.DataFrame, n: int, tasa_fraude: float) -> list[pd.Series]:
    """Muestrea n transacciones del test set respetando la tasa de fraude de la barra lateral."""
    legitimas, fraudes = df[df["Class"] == 0], df[df["Class"] == 1]
    filas = []
    for _ in range(n):
        origen = fraudes if random.random() < tasa_fraude / 100 else legitimas
        filas.append(origen.iloc[random.randrange(len(origen))])
    return filas


# ------------------------------------------------------------------ piezas visuales
def banner_alerta(reg: dict) -> None:
    st.markdown(
        f"""<div style="background:#b91c1c;color:#fff;padding:14px 20px;border-radius:10px;
        border-left:8px solid #fecaca;margin-bottom:12px;">
        <div style="font-size:1.15rem;font-weight:700;">🚨 ALERTA: transacción bloqueada por patrón atípico</div>
        <div style="margin-top:4px;font-size:1rem;">
        <b>{reg['ID']}</b> &nbsp;·&nbsp; {MONEDA}{reg['Monto']:,.2f} &nbsp;·&nbsp;
        probabilidad de fraude <b>{reg['Probabilidad']:.1%}</b> (umbral {reg['Umbral']:.2f})
        &nbsp;·&nbsp; {reg['Comercio']}, {reg['Ciudad']} &nbsp;·&nbsp; {reg['Hora']}</div></div>""",
        unsafe_allow_html=True,
    )


def kpis(h: pd.DataFrame) -> None:
    total = len(h)
    bloq = h[h["Status"] == "blocked"]
    c = st.columns(6)
    c[0].metric("Procesadas", f"{total:,}")
    c[1].metric("Aprobadas", f"{total - len(bloq):,}")
    c[2].metric("Bloqueadas", f"{len(bloq):,}")
    c[3].metric("% bloqueo", f"{(len(bloq) / total * 100) if total else 0:.1f} %")
    c[4].metric("Monto bloqueado", f"{MONEDA}{bloq['Monto'].sum():,.2f}")
    c[5].metric("Latencia promedio", f"{h['Latencia (ms)'].mean():.2f} ms" if total else "—")


def grafica_probabilidad(h: pd.DataFrame, ventana: int) -> None:
    d = h.tail(ventana).copy()
    d["n"] = range(len(h) - len(d) + 1, len(h) + 1)
    umbral = float(d["Umbral"].iloc[-1])
    base = alt.Chart(d).encode(
        x=alt.X("n:Q", title="Transacción #", axis=alt.Axis(format="d", tickMinStep=1)),
        y=alt.Y("Probabilidad:Q", title="Probabilidad de fraude", scale=alt.Scale(domain=[0, 1])),
    )
    linea = base.mark_line(color="#9aa3b2", strokeWidth=1)
    puntos = base.mark_point(filled=True, size=70, opacity=0.95).encode(
        color=alt.Color("Status:N", title="Status",
                        scale=alt.Scale(domain=["approved", "blocked"], range=[COLOR_OK, COLOR_BLOQ])),
        shape=alt.Shape("Status:N", legend=None,
                        scale=alt.Scale(domain=["approved", "blocked"], range=["circle", "diamond"])),
        tooltip=["ID", "Monto", alt.Tooltip("Probabilidad:Q", format=".4f"), "Status", "Comercio", "Ciudad"],
    )
    regla = alt.Chart(pd.DataFrame({"u": [umbral]})).mark_rule(
        color=COLOR_UMBRAL, strokeDash=[6, 4], strokeWidth=2).encode(y="u:Q")
    etiqueta = alt.Chart(pd.DataFrame({"u": [umbral], "t": [f"umbral {umbral:.2f}"]})).mark_text(
        align="left", dx=4, dy=-8, color=COLOR_UMBRAL, fontWeight="bold").encode(
        y="u:Q", x=alt.value(4), text="t:N")
    st.altair_chart((linea + puntos + regla + etiqueta).properties(height=300), width="stretch")


def tabla_en_vivo(h: pd.DataFrame, n: int) -> None:
    cols = ["Hora", "ID", "Comercio", "Ciudad", "Monto", "Probabilidad", "Status", "Latencia (ms)", "Real"]
    t = h.tail(n).iloc[::-1][cols]

    def color_fila(fila):
        if fila["Status"] == "blocked":
            return ["background-color: rgba(214,39,40,0.30); font-weight: 600"] * len(fila)
        return [""] * len(fila)

    st.dataframe(
        t.style.apply(color_fila, axis=1).format(
            {"Monto": MONEDA + "{:,.2f}", "Probabilidad": "{:.4f}", "Latencia (ms)": "{:.2f}"}),
        hide_index=True, width="stretch", height=min(38 + 35 * len(t), 560),
    )


# ------------------------------------------------------------------ app
iniciar_estado()
if not RUTA_MUESTRA.exists():
    st.error(f"No encuentro `{RUTA_MUESTRA.relative_to(RAIZ)}`. Se lo pide a Persona 1 y se pone en `artefactos/`.")
    st.stop()
muestra = cargar_muestra()

with st.sidebar:
    st.header("Simulador de flujo")
    st.radio("Fuente de la clasificación", ["API real", "Mock local"], key="fuente",
             help="«Mock local» calcula las respuestas dentro del dashboard con el mismo contrato; "
                  "sirve mientras la API o el .pth no existan.")
    if st.session_state.fuente == "API real":
        salud = comprobar_salud()
        if salud:
            st.success(f"API conectada · modelo `{salud.get('modelo')}` · umbral {salud.get('umbral')}")
        else:
            st.error(f"Sin respuesta de la API en {API}. Cambia a «Mock local» o levántala.")
    else:
        st.warning("Respuestas SIMULADAS (no hay modelo detrás).")

    tasa = st.slider("Tasa de fraude en el flujo (%)", 0.0, 50.0, 10.0, 0.5,
                     help="En la realidad es 0.17 %; se sube para que la demo muestre alertas.")
    modo = st.radio("Envío", ["Una a una", "Ráfagas"], horizontal=True)
    if modo == "Una a una":
        velocidad = st.slider("Velocidad (transacciones/s)", 0.5, 5.0, 1.0, 0.5)
        intervalo, tam_lote = 1.0 / velocidad, 1
    else:
        tam_lote = st.slider("Tamaño de la ráfaga", 5, 500, 50, 5)
        intervalo = float(st.slider("Una ráfaga cada (s)", 1, 10, 2))
    filas_tabla = st.slider("Filas en la tabla en vivo", 10, 50, 15, 5)

    c1, c2 = st.columns(2)
    if c1.button("⏸ Pausar" if st.session_state.corriendo else "▶ Iniciar", width="stretch",
                 type="primary"):
        st.session_state.corriendo = not st.session_state.corriendo
        st.session_state.error = None
        st.rerun()
    if c2.button("🗑 Reiniciar", width="stretch"):
        for k in ("historial", "contador", "corriendo", "alerta", "error", "t0"):
            st.session_state.pop(k, None)
        st.session_state.pop("historial", None)
        st.rerun()
    if st.button("💉 Inyectar fraude", width="stretch",
                 help="Fuerza UNA transacción fraudulenta del test set para la demo."):
        fraudes = muestra[muestra["Class"] == 1]
        procesar([fraudes.iloc[random.randrange(len(fraudes))]], lote=False, inyectada=True)

st.title("🛡️ Dashboard Bancario Ejecutivo – Detección de Fraude")
st.caption(f"Pasarela de pagos · Equipo 2 · fuente: **{st.session_state.fuente}** · "
           f"{'transmitiendo' if st.session_state.corriendo else 'en pausa'}")


def panel() -> None:
    """Parte que se refresca sola (fragmento): procesa un tick del flujo y repinta todo."""
    if st.session_state.corriendo:
        procesar(elegir_filas(muestra, tam_lote, tasa), lote=(modo == "Ráfagas"))

    if st.session_state.error:
        st.error(st.session_state.error)
    if st.session_state.alerta:
        banner_alerta(st.session_state.alerta)

    h = pd.DataFrame(st.session_state.historial)
    if h.empty:
        kpis(pd.DataFrame(columns=["Status", "Monto", "Latencia (ms)"]))
        st.info("Pulsa **▶ Iniciar** en la barra lateral para empezar a transmitir transacciones, "
                "o **💉 Inyectar fraude** para probar la alerta.")
        return
    kpis(h)
    st.subheader("Probabilidad de fraude en el tiempo")
    grafica_probabilidad(h, ventana=150)
    st.subheader("Últimas transacciones")
    tabla_en_vivo(h, filas_tabla)


# run_every se evalúa en cada ejecución del script, así que sigue a los controles del lateral.
st.fragment(run_every=intervalo if st.session_state.corriendo else None)(panel)()
