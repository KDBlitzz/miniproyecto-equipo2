"""Dashboard Bancario Ejecutivo (Capa 3 – Persona 4).  Ejecutar: streamlit run dashboard/app.py

Simula un flujo de transacciones (desde artefactos/muestra_test.csv que entrega Persona 1),
las envía a la API y muestra KPIs, tabla en vivo, gráfica de probabilidad y una alerta
emergente inmediata cuando una transacción resulta `blocked`.
"""
import streamlit as st

API = "http://localhost:8000/api/v1"

st.set_page_config(page_title="Dashboard Bancario – Equipo 2", layout="wide")
st.title("Dashboard Bancario Ejecutivo – Detección de Fraude")
st.info("Persona 4: pendiente")
