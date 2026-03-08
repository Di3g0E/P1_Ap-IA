import pandas as pd


def add_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    # Ordenamos por fecha y agrupamos por mes
    df = df.sort_values(by='Date').reset_index(drop=True)
    df['Month_End'] = df['Date'].dt.to_period('M').dt.to_timestamp('M')

    # Agrupación por categoría: nos dirá cuánto se gastó en Comida, Ocio, Facturas, etc., el mes anterior.
    area_monthly = df[df['Type'] == 'Expenses'].groupby(['Month_End', 'Area'])['Amount'].sum().unstack(fill_value=0)

    # Agrupación General (Income y Expenses)
    monthly = df.groupby(['Month_End', 'Type'])['Amount'].sum().unstack(fill_value=0).reset_index()
    if 'Expenses' not in monthly.columns: monthly['Expenses'] = 0
    if 'Income' not in monthly.columns: monthly['Income'] = 0

    # Fusionar con las Áreas de Gasto
    ts_df = monthly.set_index('Month_End').join(area_monthly, how='left').fillna(0).sort_index()
    ts_df.index.freq = 'ME'

    # Variables Temporales
    df_features = ts_df.copy()
    df_features['Month'] = df_features.index.month
    df_features['Quarter'] = df_features.index.quarter
    df_features['Is_Summer'] = df_features['Month'].isin([6, 7, 8, 9]).astype(int)
    df_features['Is_December'] = (df_features['Month'] == 12).astype(int)

    # Lags y Variables Rodantes Aumentadas: Lags directos de 1 a 3 meses y estacional de 12 meses (donde haya datos)
    for lag in [1, 2, 3, 12]:
        df_features[f'Expenses_Lag_{lag}'] = df_features['Expenses'].shift(lag)
        df_features[f'Income_Lag_{lag}'] = df_features['Income'].shift(lag)
        
    # Variables Relativas (Ratios y Variaciones)
    df_features['Savings_Lag_1'] = df_features['Income_Lag_1'] - df_features['Expenses_Lag_1'].abs()
    df_features['Income_Variation'] = df_features['Income_Lag_1'] - df_features['Income_Lag_2']
    df_features['Expenses_Variation'] = df_features['Expenses_Lag_1'].abs() - df_features['Expenses_Lag_2'].abs()

    # Medias Móviles y Volatilidad (Desviación Estándar)
    df_features['Expenses_Rolling_3_Mean'] = df_features['Expenses'].shift(1).rolling(window=3).mean()
    df_features['Expenses_Rolling_3_Std'] = df_features['Expenses'].shift(1).rolling(window=3).std()
    df_features['Expenses_Rolling_6_Mean'] = df_features['Expenses'].shift(1).rolling(window=6).mean()

    # Medias Móviles Exponenciales (EMA) (Da más peso a los meses recientes)
    df_features['Expenses_EMA_3'] = df_features['Expenses'].shift(1).ewm(span=3, adjust=False).mean()

    # Mover las categorías de área un paso atrás (Queremos predecir este mes usando el área del mes pasado)
    for area_col in area_monthly.columns:
        df_features[f'{area_col}_Lag_1'] = df_features[area_col].shift(1)

    df_features = df_features.drop(columns=area_monthly.columns) # Quitar el presente para no hacer trampa

    # --- ESTRATEGIA DE IMPUTACIÓN ROBUSTA ---
    # 1. Rellenar hacia atrás para mantener la tendencia inicial
    df_features = df_features.bfill() 
    
    # 2. Rellenar con la mediana de cada columna los NaNs restantes
    # Esto evita que el modelo falle sin borrar filas
    for col in df_features.columns:
        if df_features[col].isnull().any():
            median_val = df_features[col].median()
            # Si la columna es toda NaNs, la mediana será NaN, usamos 0
            df_features[col] = df_features[col].fillna(median_val if not pd.isna(median_val) else 0)

    # Aseguramos que la columna de desviación estándar no tenga NaNs (pasa en la 1ª fila)
    if 'Expenses_Rolling_3_Std' in df_features.columns:
        df_features['Expenses_Rolling_3_Std'] = df_features['Expenses_Rolling_3_Std'].fillna(0)

    if verbose:
        print("\nDatos imputados correctamente (0 NaNs generados)")
        print("Variables disponibles: ", df_features.columns.tolist())
    
    return df_features
