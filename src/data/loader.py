import pandas as pd
import os

def load_raw_data(filename: str = 'db_orig'):
    """
    Carga los datos financieros desde el archivo CSV original.
    """
    # Determinamos la raíz del proyecto (3 niveles arriba de src/data/loader.py)
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_path, "data", "raw", filename + '.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")
    
    df = pd.read_csv(file_path)
    
    return df


def load_train_test_data(filename: str = ''):
    """
    Carga los datos financieros desde el archivo CSV procesado.
    """
    # Determinamos la raíz del proyecto
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path_train = os.path.join(base_path, "data", "processed", filename + "_" + 'train.csv')
    file_path_test = os.path.join(base_path, "data", "processed", filename + "_" + 'test.csv')
    
    if not os.path.exists(file_path_train):
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path_train}")
    if not os.path.exists(file_path_test):
        raise FileNotFoundError(f"No se encontró el archivo en: {file_path_test}")
    
    df_train = pd.read_csv(file_path_train)
    df_test = pd.read_csv(file_path_test)
    
    return df_train, df_test


def split_target(df: pd.DataFrame, target: str = 'Expenses'):
    """
    Divide el DataFrame en features (X) y target (y).
    """
    FEATURES = [c for c in df.columns if c != target and c != 'Income'] # Income del mes en curso también sería trampa predecirlo

    X = df[FEATURES]
    y = df[target]

    return X, y

if __name__ == "__main__":
    # Test rápido de ejecución
    try:
        data = load_raw_data("db_orig")
        df_train, df_test = load_train_data()
        print("Datos cargados correctamente:")
        print("Datos originales:\n", data.head())
        print("Datos de entrenamiento:\n", df_train.head())
        print("Datos de prueba:\n", df_test.head())
    except Exception as e:
        print(f"Error: {e}")
