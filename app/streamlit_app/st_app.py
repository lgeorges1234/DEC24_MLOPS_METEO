from collections import OrderedDict
import streamlit as st
import requests
import os

# Importations de votre configuration
from streamlit_app import config

# Chemin du dossier des images
##IMAGE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'streamlit_app', 'Image')
IMAGE_FOLDER = os.path.join('/app', 'streamlit_app', 'Image')
# Chemin de l'API

API_URL = os.getenv('API_URL', 'http://localhost:8000')
AIRFLOW_URL = os.getenv('AIRFLOW_URL', 'http://localhost:8080')

# Créer un nouveau module de tab pour la prédiction météo
class WeatherPrediction:
    sidebar_name = "Weather Prediction"

    def run(self):
        st.title("Weather Prediction Application")

        # Fonction pour faire la prédiction manuelle
        def make_manual_prediction(input_data):
            try:
                response = requests.post(f"{API_URL}/predict_user", json=input_data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                return None

        # Fonction pour obtenir la prédiction automatique
        def get_automatic_prediction():
            try:
                # # Add basic authentication - default credentials are 'airflow'/'airflow'
                # auth = ("airflow", "airflow")  
                
                # response = requests.get(
                #     f"{AIRFLOW_URL}/api/v1/dags/3_weather_prediction_dag/dagRuns",
                #     auth=auth
                # )
                response = requests.get(f"{API_URL}/predict")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Erreur lors de la prédiction automatique : {e}")
                return None

        # Mise en page avec des onglets
        tab1, tab2 = st.tabs(["Prédiction Manuelle", "Prédiction Automatique"])

        with tab1:
            st.header("Saisie Manuelle des Données Météorologiques")
            
            # Colonnes pour la disposition des entrées
            col1, col2 = st.columns(2)
            
            with col1:
                location = st.number_input("Location Code", min_value=1, value=1)
                min_temp = st.number_input("Température Minimale", min_value=-50.0, max_value=60.0, value=10.0)
                max_temp = st.number_input("Température Maximale", min_value=-50.0, max_value=60.0, value=25.0)
                wind_gust_dir = st.number_input("Direction Rafale de Vent", min_value=0.0, max_value=360.0, value=180.0)
                wind_gust_speed = st.number_input("Vitesse Rafale de Vent", min_value=0.0, max_value=200.0, value=30.0)
            
            with col2:
                wind_dir_9am = st.number_input("Direction Vent 9h", min_value=0.0, max_value=360.0, value=180.0)
                wind_dir_3pm = st.number_input("Direction Vent 15h", min_value=0.0, max_value=360.0, value=180.0)
                wind_speed_9am = st.number_input("Vitesse Vent 9h", min_value=0.0, max_value=200.0, value=15.0)
                wind_speed_3pm = st.number_input("Vitesse Vent 15h", min_value=0.0, max_value=200.0, value=25.0)
            
            # Autres entrées
            humidity_9am = st.slider("Humidité 9h", min_value=0.0, max_value=100.0, value=70.0)
            humidity_3pm = st.slider("Humidité 15h", min_value=0.0, max_value=100.0, value=50.0)
            pressure_3pm = st.number_input("Pression à 15h", min_value=900.0, max_value=1100.0, value=1013.0)
            cloud_9am = st.slider("Couverture Nuageuse 9h", min_value=0.0, max_value=9.0, value=3.0)
            cloud_3pm = st.slider("Couverture Nuageuse 15h", min_value=0.0, max_value=9.0, value=5.0)
            rain_today = st.radio("Pluie Aujourd'hui", [0, 1])
            
            # Bouton de prédiction
            if st.button("Obtenir la Prédiction"):
                input_data = {
                    "Location": location,
                    "MinTemp": min_temp,
                    "MaxTemp": max_temp,
                    "WindGustDir": wind_gust_dir,
                    "WindGustSpeed": wind_gust_speed,
                    "WindDir9am": wind_dir_9am,
                    "WindDir3pm": wind_dir_3pm,
                    "WindSpeed9am": wind_speed_9am,
                    "WindSpeed3pm": wind_speed_3pm,
                    "Humidity9am": humidity_9am,
                    "Humidity3pm": humidity_3pm,
                    "Pressure3pm": pressure_3pm,
                    "Cloud9am": cloud_9am,
                    "Cloud3pm": cloud_3pm,
                    "RainToday": rain_today
                }
                
                prediction = make_manual_prediction(input_data)
                
                if prediction:
                    st.success(f"Prédiction : {prediction['prediction']}")
                    st.info(f"Probabilité : {prediction['probability']:.2%}")

                    # Afficher l'image correspondant à la prédiction
                    image_filename = 'rain.jpeg' if prediction['prediction'] == 0.0 else 'sun.jpeg'
                    image_path = os.path.join(IMAGE_FOLDER, image_filename)
                    
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"Prédiction : {prediction['prediction']}")
                    else:
                        st.warning(f"Image non trouvée : {image_path}")

        with tab2:
            st.header("Prédiction Automatique")
            
            if st.button("Obtenir la Prédiction Automatique"):
                prediction = get_automatic_prediction()
                
                if prediction:
                    st.success(f"Prédiction : {prediction['prediction']}")
                    st.info(f"Probabilité : {prediction['probability']:.2%}")
                    st.info(f"Message : {prediction.get('message', 'Pas de message disponible')}")

        # Pied de page
        st.markdown("---")
        st.markdown("Application de Prédiction Météorologique")

# Fichier principal (main.py)
st.set_page_config(
    page_title=config.TITLE,
    page_icon="https://datascientest.com/wp-content/uploads/2020/03/cropped-favicon-datascientest-1-32x32.png",
)

with open("streamlit_app/style.css", "r") as f:
    style = f.read()

st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)

TABS = OrderedDict(
    [
        (WeatherPrediction.sidebar_name, WeatherPrediction),
    ]
)

def run():
    st.sidebar.image(
        "https://dst-studio-template.s3.eu-west-3.amazonaws.com/logo-datascientest.png",
        width=200,
    )
    tab_name = st.sidebar.radio("", list(TABS.keys()), 0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"## {config.PROMOTION}")

    st.sidebar.markdown("### Team members :")
    for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

    tab_class = TABS[tab_name]
    # Créez une instance de la classe avant d'appeler run()
    tab = tab_class()
    tab.run()

if __name__ == "__main__":
    run()