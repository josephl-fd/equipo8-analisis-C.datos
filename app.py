
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Equipo 8 - Crédito y Ahorro", layout="wide")

st.title("Team 8: Financial Inclusion and Savings Behavior Analysis")
st.write("La idea es analizar ahorro, crédito e inclusión financiera.")

# 1. Carga de datos
df = pd.read_csv("Ahorro_Credito_Limpio.csv")

st.header("1. Revisión inicial de la base")
st.write(f"Dimensiones del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

st.subheader("Primeras 5 filas")
st.dataframe(df.head(5))

st.subheader("Últimas 5 filas")
st.dataframe(df.tail(5))

st.header("2. Información general y valores faltantes")

valores_faltantes = df.isnull().sum()
duplicados = df.duplicated().sum()

col1, col2 = st.columns(2)
col1.metric("Total de valores faltantes", int(valores_faltantes.sum()))
col2.metric("Registros duplicados", int(duplicados))

st.subheader("Valores faltantes por columna")
st.dataframe(valores_faltantes)

st.header("3. Análisis estadístico descriptivo")
st.dataframe(df.describe())

st.subheader("Medidas estadísticas por zona")
st.dataframe(df.groupby("zona")[["ingreso_mensual", "ahorro_mensual"]].mean())

st.header("4. Consistencia de datos")
st.write("Se evalúa que no existan casos donde el ahorro mensual sea mayor o igual al ingreso mensual.")
inconsistencias = (df["ahorro_mensual"] >= df["ingreso_mensual"]).sum()
st.metric("Casos con ahorro mayor o igual al ingreso", int(inconsistencias))

# Mantener la misma limpieza del Colab
df = df[df["ahorro_mensual"] < df["ingreso_mensual"]]
st.write(f"Dimensiones del dataset después de eliminar inconsistencias: {df.shape[0]} filas x {df.shape[1]} columnas")

st.header("5. Porcentaje de ahorro")
df["Porcentaje_Ahorro"] = (df["ahorro_mensual"] / df["ingreso_mensual"]) * 100
st.dataframe(df.head())

st.header("6. Acceso al crédito según departamento")
st.dataframe(df.groupby("departamento")["acceso_credito"].value_counts())

st.header("7. Gráfico 1: Ahorro mensual por tipo de institución")
fig, ax = plt.subplots(figsize=(2,1))
df.groupby("tipo_institucion")["ahorro_mensual"].sum().plot(kind="bar", ax=ax)
ax.set_title("Ahorro mensual por tipo de institución")
ax.set_xlabel("Tipo de institución")
ax.set_ylabel("Ahorro mensual")
plt.xticks(rotation=45)
st.pyplot(fig)

st.write(
    "Este gráfico permite comparar qué tipo de institución financiera está asociada "
    "con mayores niveles de ahorro mensual acumulado."
)

st.header("8. Gráfico 2: Cuenta bancaria según zona")

st.subheader("Proporción de personas que tienen cuenta bancaria por zona")
st.dataframe(df.groupby("zona")["tiene_cuenta_bancaria"].value_counts(normalize=True))

tabla_zona = pd.crosstab(df["zona"], df["tiene_cuenta_bancaria"])
st.subheader("Tabla cruzada: zona y cuenta bancaria")
st.dataframe(tabla_zona)

fig, ax = plt.subplots()
tabla_zona.plot(kind="bar", ax=ax)
ax.set_title("Tenencia de cuenta bancaria según zona")
ax.set_xlabel("Zona")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=0)
st.pyplot(fig)

st.header("9. Gráfico 3: Acceso al crédito según historial crediticio")

tabla_credito = pd.crosstab(df["historial_crediticio"], df["acceso_credito"])
st.dataframe(tabla_credito)

fig, ax = plt.subplots()
tabla_credito.plot(kind="bar", ax=ax)
ax.set_title("Acceso al crédito según historial crediticio")
ax.set_xlabel("Historial crediticio")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=0)
st.pyplot(fig)

st.write("El acceso al crédito está asociado con el historial crediticio y el uso de instituciones formales.")

st.header("10. Gráfico 4: Acceso al crédito según zona")

st.subheader("Proporción de acceso al crédito por zona")
st.dataframe(df.groupby("zona")["acceso_credito"].value_counts(normalize=True))

tabla_credito_segun_zona = pd.crosstab(df["zona"], df["acceso_credito"])
st.dataframe(tabla_credito_segun_zona)

fig, ax = plt.subplots(figsize=(6,4))
tabla_credito_segun_zona.plot(kind="bar", ax=ax)
ax.set_title("Acceso al crédito según zona")
ax.set_xlabel("Zona")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=0)
st.pyplot(fig)

st.header("11. Gráfico 5: Scatter ingreso vs ahorro por zona")

colores = df["zona"].map({"Urbana": "blue", "Rural": "red"})
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df["ingreso_mensual"], df["ahorro_mensual"], c=colores)
ax.set_xlabel("Ingreso")
ax.set_ylabel("Ahorro")
ax.set_title("Ingreso vs Ahorro por zona")
st.pyplot(fig)

st.write(
    "Se observa la relación entre ingreso y ahorro, diferenciando las zonas urbana y rural."
)

st.header("12. Gráfico 6: Histograma de ingresos mensuales y ahorros mensuales")

fig, ax = plt.subplots(figsize=(6,4))
df["ingreso_mensual"].plot(kind="hist", color="blue", ax=ax)
ax.set_title("Distribución de ingresos")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6,4))
df["ahorro_mensual"].plot(kind="hist", color="red", ax=ax)
ax.set_title("Distribución de ahorros")
st.pyplot(fig)

st.header("13. Clasificación de inclusión financiera")

def puntos_si_no(valor):
    if valor == "Sí":
        return 1
    else:
        return 0

df["Puntos_Cuenta"] = df["tiene_cuenta_bancaria"].apply(puntos_si_no)
df["Puntos_Credito"] = df["acceso_credito"].apply(puntos_si_no)

def puntos_historial(valor):
    if valor == "Bueno":
        return 2
    elif valor == "Regular":
        return 1
    elif valor == "Malo":
        return 0.5
    else:
        return 0

df["Puntos_Historial"] = df["historial_crediticio"].apply(puntos_historial)

df["Porcentaje_Ahorro"] = df["ahorro_mensual"] / df["ingreso_mensual"] * 100
df["Porcentaje_Ahorro"] = df["Porcentaje_Ahorro"].fillna(0)

q1 = df["Porcentaje_Ahorro"].quantile(0.25)
q3 = df["Porcentaje_Ahorro"].quantile(0.75)

def puntos_ahorro(valor):
    if valor >= q3:
        return 2
    elif valor >= q1:
        return 1
    else:
        return 0

df["Puntos_Ahorro"] = df["Porcentaje_Ahorro"].apply(puntos_ahorro)

q1_ingreso = df["ingreso_mensual"].quantile(0.25)
q3_ingreso = df["ingreso_mensual"].quantile(0.75)

def puntos_ingreso(valor):
    if valor >= q3_ingreso:
        return 2
    elif valor >= q1_ingreso:
        return 1
    else:
        return 0

df["Puntos_Ingreso"] = df["ingreso_mensual"].apply(puntos_ingreso)

df["Puntaje_Inclusion"] = (
    df["Puntos_Cuenta"] +
    df["Puntos_Credito"] +
    df["Puntos_Historial"] +
    df["Puntos_Ahorro"] +
    df["Puntos_Ingreso"]
)

def clasificar_inclusion(valor):
    if valor >= 5:
        return "Alta inclusión financiera"
    elif valor >= 4:
        return "Moderada inclusión"
    else:
        return "Exclusión financiera"

df["Nivel_Inclusion"] = df["Puntaje_Inclusion"].apply(clasificar_inclusion)

st.subheader("Vista de puntajes")
st.dataframe(df.head())

st.subheader("Resumen del puntaje de inclusión")
st.dataframe(df["Puntaje_Inclusion"].describe())

st.subheader("Conteo por nivel de inclusión")
st.dataframe(df["Nivel_Inclusion"].value_counts())

st.header("14. Gráfico 7: Distribución por nivel de inclusión financiera")

fig, ax = plt.subplots(figsize=(6,4))
df["Nivel_Inclusion"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Clasificación de inclusión financiera")
ax.set_xlabel("Nivel de inclusión")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=45)
st.pyplot(fig)

st.write(
    "La clasificación resume el nivel de inclusión financiera de cada persona "
    "a partir de variables como acceso a servicios financieros, crédito, ahorro e ingreso."
)

st.header("15. Ranking de departamentos con mayor exclusión financiera")

exclusion = df[df["Nivel_Inclusion"] == "Exclusión financiera"]
ranking_exclusion = exclusion["departamento"].value_counts()

st.dataframe(ranking_exclusion.head(10))

st.header("16. Gráfico 8: Concentración de la exclusión financiera en departamentos seleccionados")

fig, ax = plt.subplots(figsize=(6,4))
ranking_exclusion.plot(kind="bar", ax=ax)
ax.set_title("Departamentos con mayor exclusión financiera")
ax.set_xlabel("Departamento")
ax.set_ylabel("Cantidad de personas")
plt.xticks(rotation=45)
st.pyplot(fig)

st.header("17. Análisis adicional")

st.subheader("Ingreso mensual promedio según nivel de inclusión financiera")
st.dataframe(df.groupby("Nivel_Inclusion")["ingreso_mensual"].mean())

st.header("18. Gráfico 9: Ingreso mensual promedio según nivel de inclusión financiera")

fig, ax = plt.subplots(figsize=(6,4))
df.groupby("Nivel_Inclusion")["ingreso_mensual"].mean().plot(kind="bar", ax=ax)
ax.set_title("Ingreso mensual promedio por nivel de inclusión financiera")
ax.set_xlabel("Nivel de inclusión financiera")
ax.set_ylabel("Ingreso mensual promedio")
plt.xticks(rotation=45)
st.pyplot(fig)

st.write(
    "Se observa una relación positiva entre ingreso mensual promedio e inclusión financiera."
)

st.subheader("Proporción de nivel de inclusión según zona")
st.dataframe(df.groupby("zona")["Nivel_Inclusion"].value_counts(normalize=True))

st.header("19. Gráfico 10: Nivel de inclusión según zona")

fig, ax = plt.subplots(figsize=(6,4))
df.groupby("zona")["Nivel_Inclusion"].value_counts(normalize=True).unstack().plot(kind="bar", ax=ax)
ax.set_title("Proporción de inclusión financiera por zona")
ax.set_ylabel("Proporción")
ax.set_xlabel("Zona")
plt.xticks(rotation=0)
st.pyplot(fig)

st.header("20. Descargar base final")

csv_final = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar CSV con puntajes y clasificación",
    data=csv_final,
    file_name="Ahorro_Credito_Clasificado.csv",
    mime="text/csv"
)
