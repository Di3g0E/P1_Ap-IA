import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

def evaluate_predictions(y_true, y_pred, model_name="Modelo"):
    """
    Calcula y devuelve un diccionario con las métricas de evaluación estándar.
    """
    # Aseguramos que no haya NaNs en las predicciones
    y_pred = np.nan_to_num(y_pred)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4)
    }
    
    return metrics

def evaluate_arima(y_true, model):
    # Pronosticar tantos días hacia adelante como tenga el set de validación
    forecast_steps = len(y_true)
    y_pred = model.forecast(steps=forecast_steps)

    # Alinear los índices al set de validación
    y_pred.index = y_true.index

    return y_pred

def evaluate_sarimax_wfv(X_train, y_train, X_test, y_test, model_fit, use_log=True):
    """
    Evaluación Walk-Forward para SARIMAX. Re-entrena el modelo en cada paso de tiempo.
    """
    # 1. Recuperar parámetros del modelo original
    order = model_fit.model.order
    seasonal_order = model_fit.model.seasonal_order
    
    # 2. Preparar historia (usando datos absolutos para evitar problemas con logs)
    history_y = y_train.abs().tolist()
    history_exog = X_train.copy()
    
    test_y = y_test.abs().tolist()
    test_exog = X_test.copy()
    
    # Aplicar logaritmos si el modelo original los usó
    if use_log:
        history_y = np.log1p(history_y).tolist()
        test_y_log = np.log1p(test_y).tolist()
        for col in history_exog.columns:
            if history_exog[col].abs().max() > 1:
                history_exog[col] = np.log1p(history_exog[col].abs())
                test_exog[col] = np.log1p(test_exog[col].abs())
    else:
        test_y_log = test_y

    predictions = []
    for t in range(len(y_test)):
        # Ajustar modelo con historia actual
        model = SARIMAX(endog=history_y, exog=history_exog,
                        order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        
        current_fit = model.fit(disp=False)
        
        # Predecir siguiente paso usando las exógenas de ese momento
        next_exog = test_exog.iloc[[t]]
        yhat_log = current_fit.forecast(steps=1, exog=next_exog).iloc[0]
        
        # Guardar predicción (revertir logaritmo y RECUPERAR EL SIGNO ORIGINAL)
        # Como los gastos (y) vienen en negativo, la media de y_train nos indica el signo.
        target_sign = 1 if y_train.mean() >= 0 else -1
        
        res = np.expm1(yhat_log) if use_log else yhat_log
        res = res * target_sign # <-- CORRECCIÓN CLAVE
        
        predictions.append(res)
        
        # Actualizar historia con el dato real del test para el siguiente paso
        history_y.append(test_y_log[t])
        history_exog = pd.concat([history_exog, test_exog.iloc[[t]]])

    return pd.Series(predictions, index=y_test.index)

def evaluate_lstm(X_test, y_test, model_lstm, scaler_X, scaler_y, time_steps=3):
    """
    Evalúa el modelo LSTM generando las secuencias de test y desescalando las predicciones.
    """
    # 1. Escalar características
    scaled_test = scaler_X.transform(X_test)
    
    # 2. Secuencias
    def create_test_sequences(features, ts):
        X_seq = []
        for i in range(len(features) - ts):
            X_seq.append(features[i:(i + ts)])
        return np.array(X_seq)
        
    X_test_seq = create_test_sequences(scaled_test, time_steps)
    
    # Si las secuencias de test están vacías (X_test es menor que time_steps)
    if len(X_test_seq) == 0:
        print("⚠️ Advertencia: No hay suficientes datos de prueba para evaluar LSTM.")
        return y_test.iloc[[]], pd.Series([], dtype=float)
    
    # 3. Predecir
    y_pred_scaled = model_lstm.predict(X_test_seq, verbose=0)
    
    # 4. Desescalar
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    
    # Al crear secuencias perdemos los primeros 'time_steps' datos del inicio
    y_test_aligned = y_test.iloc[time_steps:].copy()
    
    return y_test_aligned, pd.Series(y_pred, index=y_test_aligned.index)

def compare_models(results_list):
    """
    Toma una lista de diccionarios de métricas y devuelve un DataFrame comparativo.
    """
    df_results = pd.DataFrame(results_list)
    return df_results.sort_values(by="MAE")
