import pandas as pd
import os

def load_raw_data(file_path="data/raw/db.csv"):
    """
    Carga los datos financieros desde el archivo CSV original.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")
    
    # Nota: El archivo parece usar comas para decimales en Amount
    df = pd.read_csv(file_path, decimal=',')
    
    # Limpieza básica inicial
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    return df

if __name__ == "__main__":
    # Test rápido de ejecución
    try:
        data = load_raw_data("../../data/raw/db.csv")
        print("Datos cargados correctamente:")
        print(data.head())
    except Exception as e:
        print(f"Error: {e}")
