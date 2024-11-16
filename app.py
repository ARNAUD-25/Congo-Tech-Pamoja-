import streamlit as st
import requests
from PIL import Image

st.title("Classification des Maladies du Riz")

uploaded_file = st.file_uploader("Choisissez une image de feuille de riz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)
    
    image_bytes = uploaded_file.read()


    if st.button("Diagnostiquer"):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                files={"file": image_bytes}
            )
            response.raise_for_status()  
            result = response.json()
            class_name = result["class_name"]
            
            st.write(f"Résultat : **{class_name}**")
        
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête : {e}")

