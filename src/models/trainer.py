import os
# Desactivar logs informativos de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
# Desactivar advertencias de Pandas y Statsmodels
warnings.filterwarnings('ignore')

import joblib
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

# Librerías para Series Temporales y Deep Learning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except ImportError:
    tf = None


def train_sklearn_model(X, y, model_type="lr", params=None, cv_splits=3):
    """
    Entrena modelos basados en Sklearn (LR, RF, HistGB, XGB) con búsqueda de hiperparámetros.
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    
    if model_type == "lr":
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    elif model_type == "rf":
        base_model = RandomForestRegressor(random_state=42)
        param_grid = params if params else {'n_estimators': [100, 200], 'max_depth': [3, 5]}
        
    elif model_type == "xgb":
        base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = params if params else {
            'n_estimators': [100, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [2, 3, 5]
        }
        
    elif model_type == "hgb":
        base_model = HistGradientBoostingRegressor(random_state=42)
        param_grid = params if params else {'learning_rate': [0.01, 0.05], 'max_iter': [100, 300], 'max_depth': [3, 5]}
    
    # Busqueda de hiperparametros
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, 
                               cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

def train_arima_model(y, order=(1, 0, 1)):
    """
    Entrena un modelo ARIMA sobre una serie temporal (y).
    """
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    return model_fit

def train_sarimax_model(X, y, seasonal=False, m=6, use_log=False):
    """
    Entrena un modelo SARIMAX encontrando los parámetros óptimos con auto_arima 
    y utilizando TODAS las variables en X como exógenas.
    """
    # 1. Preparar datos
    y_train = y.abs()
    exog_train = X.copy()
    
    if use_log:
        y_train = np.log1p(y_train)
        # Aplicamos log a las columnas numéricas que no sean binarias/pequeñas
        for col in exog_train.columns:
            if exog_train[col].abs().max() > 1:
                exog_train[col] = np.log1p(exog_train[col].abs())

    # 2. Búsqueda de parámetros óptimos
    auto_model = auto_arima(y_train, X=exog_train, seasonal=seasonal, m=m,
                            start_p=0, start_q=0, max_p=3, max_q=3,
                            d=None, D=None, trace=False,
                            error_action='ignore', suppress_warnings=True, stepwise=True)
    
    # 3. Entrenar modelo final con los órdenes encontrados
    final_model = SARIMAX(y_train, 
                          exog=exog_train, 
                          order=auto_model.order, 
                          seasonal_order=auto_model.seasonal_order,
                          enforce_stationarity=False, 
                          enforce_invertibility=False)
    
    return final_model.fit(disp=False)

def train_lstm_model(X, y, time_steps=3, epochs=150, batch_size=4):
    """
    Entrena una red neuronal profunda LSTM.
    """
    if tf is None:
        raise ImportError("TensorFlow no está instalado.")

    # 1. Escalado de la entrada completa y el target
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    # Escalar TODO el DataFrame X
    scaled_features = scaler_X.fit_transform(X)
    
    # Escalar y (necesita reshape porque es una serie 1D)
    scaled_target = scaler_y.fit_transform(y.values.reshape(-1, 1))

    # 2. Creación de secuencias
    def create_sequences(features, target, ts):
        X_seq, y_seq = [], []
        for i in range(len(features) - ts):
            X_seq.append(features[i:(i + ts)])
            y_seq.append(target[i + ts])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(scaled_features, scaled_target, time_steps)

    # 3. Arquitectura del modelo
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model.fit(X_seq, y_seq, epochs=epochs, batch_size=batch_size, 
              validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    return model, (scaler_X, scaler_y)

def train_voting_ensemble(estimators, X, y):
    """
    Crea un meta-modelo de ensamble basado en una lista de estimadores entrenados.
    """
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(X, y)
    return ensemble

def save_model(model, filename):
    """
    Guarda el modelo. Si es Keras lo guarda como .h5, si es Sklearn como .pkl.
    """
    # Determinamos la raíz del proyecto
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(base_path, "models")
    os.makedirs(save_dir, exist_ok=True)
    
    # Caso especial para modelos de Keras/TensorFlow
    if "keras" in str(type(model)):
        save_path = os.path.join(save_dir, filename + ".h5")
        model.save(save_path)
    else:
        if not filename.endswith('.pkl'): filename += '.pkl'
        save_path = os.path.join(save_dir, filename)
        joblib.dump(model, save_path)
    
    print(f"Modelo guardado en: {save_path}")
    return save_path

def load_model(filename):
    """
    Carga un modelo desde la carpeta 'models/'.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(base_path, "models", filename)
    
    if filename.endswith('.h5'):
        return tf.keras.models.load_model(save_path)
    
    if not os.path.exists(save_path):
        if not save_path.endswith('.pkl'): save_path += '.pkl'
        
    return joblib.load(save_path)
