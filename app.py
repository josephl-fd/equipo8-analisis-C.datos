
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Equipo 8 - Crédito y Ahorro", layout="wide")

st.title("Equipo 8: Acceso al crédito y ahorro financiero")
st.write("Aplicación web para analizar ahorro, crédito e inclusión financiera usando la base limpia del Equipo 8.")

df = pd.read_csv("Ahorro_Credito_Limpio.csv")

st.header("1. Vista general de la base de datos")
col1, col2, col3 = st.columns(3)
col1.metric("Registros", len(df))
col2.metric("Departamentos", df["departamento"].nunique())
col3.metric("Zonas", df["zona"].nunique())

st.subheader("Primeras filas")
st.dataframe(df.head())

st.subheader("Resumen estadístico")
st.dataframe(df.describe())

st.header("2. Verificación de limpieza")
inconsistencias = df[df["ahorro_mensual"] > df["ingreso_mensual"]]
st.write("Se verifica si existen registros donde el ahorro mensual supera el ingreso mensual.")

col1, col2 = st.columns(2)
col1.metric("Casos inconsistentes", len(inconsistencias))
col2.metric("Base limpia", "Sí" if len(inconsistencias) == 0 else "No")

if len(inconsistencias) == 0:
    st.success("No existen casos donde el ahorro mensual sea mayor al ingreso mensual.")
else:
    st.warning("Aún existen inconsistencias en la base.")
    st.dataframe(inconsistencias)

st.header("3. Creación de variables")

if "porcentaje_ahorro" not in df.columns:
    df["porcentaje_ahorro"] = (df["ahorro_mensual"] / df["ingreso_mensual"]) * 100
    df["porcentaje_ahorro"] = df["porcentaje_ahorro"].fillna(0)

st.write("El porcentaje de ahorro mide qué parte del ingreso mensual se destina al ahorro.")
st.dataframe(df[["ingreso_mensual", "ahorro_mensual", "porcentaje_ahorro"]].head())

st.header("4. Clasificación de inclusión financiera")

df["puntos_cuenta"] = df["tiene_cuenta_bancaria"].apply(lambda x: 2 if x == "Sí" else 0)
df["puntos_credito"] = df["acceso_credito"].apply(lambda x: 2 if x == "Sí" else 0)

def puntos_historial(valor):
    if valor == "Bueno":
        return 2
    elif valor == "Regular":
        return 1
    else:
        return 0

df["puntos_historial"] = df["historial_crediticio"].apply(puntos_historial)

q1_ahorro = df["porcentaje_ahorro"].quantile(0.33)
q2_ahorro = df["porcentaje_ahorro"].quantile(0.66)

def puntos_ahorro(valor):
    if valor >= q2_ahorro:
        return 2
    elif valor >= q1_ahorro:
        return 1
    else:
        return 0

df["puntos_ahorro"] = df["porcentaje_ahorro"].apply(puntos_ahorro)

q1_ingreso = df["ingreso_mensual"].quantile(0.33)
q2_ingreso = df["ingreso_mensual"].quantile(0.66)

def puntos_ingreso(valor):
    if valor >= q2_ingreso:
        return 2
    elif valor >= q1_ingreso:
        return 1
    else:
        return 0

df["puntos_ingreso"] = df["ingreso_mensual"].apply(puntos_ingreso)

df["puntaje_inclusion"] = (
    df["puntos_cuenta"] +
    df["puntos_credito"] +
    df["puntos_historial"] +
    df["puntos_ahorro"] +
    df["puntos_ingreso"]
)

def clasificar_inclusion(valor):
    if valor >= 8:
        return "Alta inclusión financiera"
    elif valor >= 4:
        return "Moderada inclusión"
    else:
        return "Exclusión financiera"

df["nivel_inclusion"] = df["puntaje_inclusion"].apply(clasificar_inclusion)

st.write("Se asignaron puntajes según cuenta bancaria, crédito, historial, ahorro e ingreso.")
st.dataframe(df[["ingreso_mensual", "ahorro_mensual", "porcentaje_ahorro", "puntaje_inclusion", "nivel_inclusion"]].head())

st.header("5. Visualizaciones principales")

st.subheader("Ahorro mensual promedio por tipo de institución")
ahorro_institucion = df.groupby("tipo_institucion")["ahorro_mensual"].mean()
fig, ax = plt.subplots()
ahorro_institucion.plot(kind="bar", ax=ax)
ax.set_title("Ahorro mensual promedio por tipo de institución")
ax.set_xlabel("Tipo de institución")
ax.set_ylabel("Ahorro mensual promedio")
plt.xticks(rotation=45)
st.pyplot(fig)
st.write("Se usa el promedio para comparar mejor el comportamiento de ahorro por persona.")

st.subheader("Proporción de personas con cuenta bancaria según zona")
tabla_zona = pd.crosstab(df["zona"], df["tiene_cuenta_bancaria"], normalize="index")
fig, ax = plt.subplots()
tabla_zona.plot(kind="bar", ax=ax)
ax.set_title("Proporción de cuenta bancaria por zona")
ax.set_xlabel("Zona")
ax.set_ylabel("Proporción")
plt.xticks(rotation=0)
st.pyplot(fig)
st.dataframe(tabla_zona)

st.subheader("Acceso al crédito según historial crediticio")
tabla_credito = pd.crosstab(df["historial_crediticio"], df["acceso_credito"], normalize="index")
fig, ax = plt.subplots()
tabla_credito.plot(kind="bar", ax=ax)
ax.set_title("Acceso al crédito según historial crediticio")
ax.set_xlabel("Historial crediticio")
ax.set_ylabel("Proporción")
plt.xticks(rotation=0)
st.pyplot(fig)
st.dataframe(tabla_credito)

st.subheader("Distribución del nivel de inclusión financiera")
fig, ax = plt.subplots()
df["nivel_inclusion"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Nivel de inclusión financiera")
ax.set_xlabel("Nivel de inclusión")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=45)
st.pyplot(fig)

st.header("6. Ranking de departamentos con mayor exclusión financiera")
ranking = (
    df[df["nivel_inclusion"] == "Exclusión financiera"]
    .groupby("departamento")
    .size()
    .sort_values(ascending=False)
)

st.dataframe(ranking.head(10))

fig, ax = plt.subplots()
ranking.head(10).plot(kind="bar", ax=ax)
ax.set_title("Top 10 departamentos con mayor exclusión financiera")
ax.set_xlabel("Departamento")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=45)
st.pyplot(fig)

st.header("7. Análisis adicional")

st.subheader("Ingreso mensual promedio por nivel de inclusión financiera")
ingreso_inclusion = df.groupby("nivel_inclusion")["ingreso_mensual"].mean()
fig, ax = plt.subplots()
ingreso_inclusion.plot(kind="bar", ax=ax)
ax.set_title("Ingreso mensual promedio por nivel de inclusión financiera")
ax.set_xlabel("Nivel de inclusión financiera")
ax.set_ylabel("Ingreso mensual promedio")
plt.xticks(rotation=45)
st.pyplot(fig)

st.write("Este análisis permite observar si el ingreso cambia según el nivel de inclusión financiera.")

st.subheader("Inclusión financiera por zona")
tabla_inclusion_zona = pd.crosstab(df["zona"], df["nivel_inclusion"], normalize="index")
fig, ax = plt.subplots()
tabla_inclusion_zona.plot(kind="bar", ax=ax)
ax.set_title("Proporción de inclusión financiera por zona")
ax.set_xlabel("Zona")
ax.set_ylabel("Proporción")
plt.xticks(rotation=0)
st.pyplot(fig)
st.dataframe(tabla_inclusion_zona)

st.header("8. Descargar base con clasificación")
csv_final = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar CSV con clasificación",
    data=csv_final,
    file_name="Ahorro_Credito_Clasificado.csv",
    mime="text/csv"
)
