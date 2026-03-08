import os
import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_split(df: pd.DataFrame, dataset_name: str, test_size: float = 0.3, val_size: float = 0.5, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide un DataFrame en train, validation y test guardándolos en data/processed.
    """
    # 1. Dividir datos (manteniendo orden temporal si es necesario)
    train_df, temp_df = train_test_split(df, test_size=test_size, shuffle=False)
    if val_size > 0:
        val_df, test_df = train_test_split(temp_df, test_size=val_size, shuffle=False)
    else:
        val_df = pd.DataFrame()
        test_df = temp_df

    # 2. Determinar ruta de salida (data/processed/ en la raíz del proyecto)
    # __file__ es '.../src/data/preparation.py', subimos 3 niveles para llegar a la raíz del proyecto
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_path, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Guardar datos
    dataset_name = dataset_name + "_"
    train_df.to_csv(os.path.join(output_dir, f"{dataset_name}train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, f"{dataset_name}val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{dataset_name}test.csv"), index=False)
    
    if verbose:
        print(f"Archivos guardados en: {output_dir}")
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def preprocess_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    # 1. Convertimos 'Date' a datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # 2. 'Amount' viene como string con coma, € y punto. Lo pasamos a float
    # Verificamos si es tipo objeto (string en pandas)
    if df['Amount'].dtype == 'str':
        df['Amount'] = df['Amount'].astype(str).str.replace('€', '', regex=False)
        df['Amount'] = df['Amount'].str.replace('.', '', regex=False)
        df['Amount'] = df['Amount'].str.replace(',', '.', regex=False).astype(float)

    # 3. Eliminar Outliers / Ruido (según cuaderno de pruebas)
    outlier_categories = ['Leisure, Vacations', 'Invoice']
    df = df[~df['Area'].isin(outlier_categories)].copy()

    # 4. Convertir los gastos ('Expenses') en valores negativos
    if 'Type' in df.columns:
        df.loc[df['Type'] == 'Expenses', 'Amount'] = -df.loc[df['Type'] == 'Expenses', 'Amount'].abs()

    # 5. Normalizar Nombres de Área (Codificación corta)
    def get_code(name):
        words = str(name).replace(",", "").split()
        return "".join([w[:2].capitalize() for w in words])

    if verbose:
        mapping = ", ".join([f"{a} -> {get_code(a)}" for a in df['Area'].unique()])
        print(f"Mapeo de áreas: {mapping}")

    df['Area'] = df['Area'].apply(get_code)

    return df
