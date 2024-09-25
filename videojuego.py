import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import *
import pickle
import streamlit as st
import joblib
import requests


url_model = "https://raw.githubusercontent.com/minivillalba4/videogames/main/models/model_videogame%20(2).pkl"
url_scaler_X = "https://raw.githubusercontent.com/minivillalba4/videogames/main/models/scaler_x%20(1).pkl"
url_target_encoder = "https://raw.githubusercontent.com/minivillalba4/videogames/main/models/target_encoder_videogame%20(1).pkl"
url_explainer="https://raw.githubusercontent.com/minivillalba4/videogames/main/models/explainer.pkl"
url_shap_values="https://raw.githubusercontent.com/minivillalba4/videogames/main/models/shap_values.pkl"


# Descargar los archivos
response_model = requests.get(url_model)
response_scaler_X = requests.get(url_scaler_X)
response_target_encoder = requests.get(url_target_encoder)
response_explainer = requests.get(url_explainer)
response_shap_values = requests.get(url_shap_values)



# Guardar los archivos localmente
open("model_videogame.pkl", "wb").write(response_model.content)
open("scaler_X.pkl", "wb").write(response_scaler_X.content)
open("target_encoder.pkl", "wb").write(response_target_encoder.content)

# Carga de modelos y transformadores
model = joblib.load("model_videogame.pkl")
scaler_X = joblib.load("scaler_X.pkl")
target_encoder = joblib.load("target_encoder.pkl")


#Importar el df original
df=pd.read_csv("https://raw.githubusercontent.com/minivillalba4/videogames/main/online_gaming_behavior_dataset.csv")

#Preguntar al usuario
edad=st.sidebar.slider("Edad del usuario",1.0,140.0,df["Age"].mean(),1.0)
sexo = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
sexo_dict = {"Hombre": [1, 0], "Mujer": [0, 1]}
sexo_values = sexo_dict[sexo]
sexo_df = pd.DataFrame([sexo_values], columns=["Gender_Male", "Gender_Female"])
sexo_df.drop(columns=["Gender_Female"],axis=1,inplace=True)
localizacion=st.sidebar.selectbox("Localizacion",['Other', 'USA', 'Europe', 'Asia'])
genero_videojuego=st.sidebar.selectbox("Género del videojuego",['Estrategia', 'Deportes', 'Acción', 'RPG', 'Simulación'])
horas_juego=st.sidebar.slider("Horas de Juego",1.0,df["PlayTimeHours"].max()+10.0,df["PlayTimeHours"].mean(),1.0)
compras_en_videojuegos=st.sidebar.radio("Ha realizado compras en videojuegos",["No","Si"])
dificultad_videojuego=st.sidebar._select_slider("Localizacion",['Facil', 'Normal', 'Difícil'])
sesiones_semanales=st.sidebar.slider("Sesiones por semana",0.0,df["PlayTimeHours"].max()+7.0,df["PlayTimeHours"].mean(),1.0)
duracion_sesion=st.sidebar.slider("Duracion de la sesion",1.0,df["AvgSessionDurationMinutes"].max()+7.0,df["AvgSessionDurationMinutes"].mean(),1.0)
nivel_jugador=st.sidebar.slider("Nivel del jugador",1.0,df["PlayerLevel"].max()+7.0,df["PlayerLevel"].mean(),1.0)
logros_desbloqueados=st.sidebar.slider("Logros desbloqueados",0.0,df["AchievementsUnlocked"].max()+7.0,df["AchievementsUnlocked"].mean(),1.0)


features={
    'Age':edad,
    'Location':localizacion,
    'GameGenre':genero_videojuego,
    'PlayTimeHours':horas_juego,
    'InGamePurchases':compras_en_videojuegos,
    'GameDifficulty':dificultad_videojuego,
    'SessionsPerWeek':sesiones_semanales,
    'AvgSessionDurationMinutes':duracion_sesion,
    'PlayerLevel':nivel_jugador,
    'AchievementsUnlocked':logros_desbloqueados,
}
data=pd.DataFrame(features,index=[0])
data = pd.concat([data, sexo_df], axis=1)

st.title("Prediccion modelo videojuegos")
if st.checkbox("Mostrar dimensión del conjunto de datos"):
    st.text("Dimensión del conjunto de datos:")
    st.text(df.shape)
if st.checkbox("Mostrar conjunto de datos"):
    st.dataframe(df.head())
if st.checkbox("Mostrar estadísticos principales del conjunto de datos"):
    st.dataframe(df.describe())
#Transformación de los datos
df["EngagementLevel"]=df["EngagementLevel"].map({"Low":0,"Medium":1,"Hard":2})
data["GameDifficulty"]=data["GameDifficulty"].map({"Facil":0,"Normal":1,"Difícil":2})
data["InGamePurchases"]=data["InGamePurchases"].map({"No":0,"Si":1})

#Targer encoder
data=target_encoder.transform(data)

#Reordenar
expected_order=scaler_X.feature_names_in_
data=data[list(expected_order)]

#Escalador
data_sc=scaler_X.transform(data)

st.subheader("Prediccion")
if st.checkbox("Mostrar predicción"):
    st.write("La predicción es")
    st.dataframe(model.predict(data_sc))


open("explainer.pkl", "wb").write(response_explainer.content)
open("shap_values.pkl", "wb").write(response_shap_values.content)
explainer =joblib.load("explainer.pkl")
shap_values =joblib.load("shap_values.pkl")

st.subheader("Importancia de las características")
if st.checkbox("Mostrar importancia de las características"):
    
    clase=1
    expected_value= np.mean(model.predict_proba(X_train)[:, clase])
    
    obs_force = 2
    shap.initjs()
    shap.force_plot(expected_value, shap_values[obs_force,:,1], X_test.iloc[obs_force, :])
    
