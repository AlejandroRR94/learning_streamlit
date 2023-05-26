import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('models/decision_tree_model.pkl')

def cargar_datos(archivo):
    # Cargar los datos del archivo en un DataFrame de pandas
    datos = pd.read_csv(archivo)
    return datos

def predecir(datos):
    # Realizar las predicciones utilizando el modelo cargado
    predicciones = model.predict(datos)
    return predicciones

def main():
    # Configuración de la aplicación Streamlit
    st.title('Aplicación de predicción')
    st.write('Arrastra y suelta tu archivo CSV para obtener predicciones.')

    # Arrastrar y soltar el archivo
    archivo = st.file_uploader('Cargar archivo CSV', type='csv')

    if archivo is not None:
        # Cargar y mostrar los datos
        datos = cargar_datos(archivo)
        st.write('Datos cargados:')
        st.write(datos)

        # Realizar las predicciones
        predicciones = predecir(datos)
        st.write('Predicciones:')
        st.write(predicciones)

# Ejecutar la aplicación
if __name__ == '__main__':
    main()
