import os
import pandas as pd



def cambiar_extension_archivos_con_nombre_original(directorio):
    for archivo in os.listdir(directorio):
        if archivo.endswith('.xlsx'):
            ruta_completa = os.path.join(directorio, archivo)
            nuevo_nombre = archivo.replace('.xlsx', '.csv')
            nueva_ruta = os.path.join(directorio, nuevo_nombre)
            
            # Leer el archivo .xlsx y guardarlo como .csv
            df = pd.read_excel(ruta_completa)
            df.to_csv(nueva_ruta, index=False)
            print(f'Archivo convertido: {ruta_completa} -> {nueva_ruta}')
            
            # Eliminar el archivo .xlsx original
            os.remove(ruta_completa)
            print(f'Archivo original eliminado: {ruta_completa}')

# Especifica el directorio que contiene los archivos .xlsx
directorio = (
    r'C:\Users\alvar\Desktop\Programacion (machine learning, data science)'
    r'\modelo pastas peine\datos\resultados_meteo\evap'
)
cambiar_extension_archivos_con_nombre_original(directorio)

