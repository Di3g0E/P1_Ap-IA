import os
import logging
import warnings

# ==========================================
# DESACTIVAR TODOS LOS WARNINGS Y LOGS
# ==========================================
# 1. TensorFlow C++ Logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 2. TensorFlow Python Logs
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_WARNINGS"] = "FALSE"
# 3. Pandas / Statsmodels Warnings
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings('ignore')

import argparse
from src.data.loader import load_raw_data, load_train_test_data, split_target
from src.data.preparation import train_val_test_split, preprocess_data
from src.features.builder import add_features
from src.utils.plots import plot_correlation_matrix
from src.models.trainer import train_sklearn_model, train_arima_model, train_sarimax_model, train_lstm_model, train_voting_ensemble, save_model
from src.evaluation.metrics import evaluate_predictions, evaluate_arima, evaluate_sarimax_wfv, evaluate_lstm
from src.visualization.plots import (
    plot_model_comparison_metrics,
    plot_actual_vs_predicted,
    plot_actual_vs_predicted_multiple,
    plot_error_distribution,
    plot_feature_importance
)
import pandas as pd

def main(args):    
    # 1. Cargar
    df_raw = load_raw_data(args.dataset)
    
    # 2. Dividir y Guardar
    train, _, test = train_val_test_split(df_raw, dataset_name=args.dataset_name, test_size=0.2, val_size=0, verbose=False)
    
    # 3. Cargar datos procesados
    train_df, _ = load_train_test_data(filename=args.dataset_name)

    # 4. Preprocesar
    train_df = preprocess_data(train_df, verbose=False)
    train_df = add_features(train_df, verbose=False)

    print("\n\n¡Datos listos para entrenar!\n\n")

    # 5. Visualizar
    # plot_correlation_matrix(train_df, save="correlation_matrix.png", view=False, verbose=False)

    # 6. Separar target
    X, y = split_target(train_df)


    # 1. ENTRENAR (Baseline y Avanzado)
    model_lr = train_sklearn_model(X, y, model_type="lr")
    model_rf = train_sklearn_model(X, y, model_type="rf")
    model_xgb = train_sklearn_model(X, y, model_type="xgb")
    model_hgb = train_sklearn_model(X, y, model_type="hgb")
    model_arima = train_arima_model(y)

    # Seleccionamos variables externas verdaderas que ayuden a predecir, descartando lags de Expenses
    columnas_externas = ['Income_Lag_1', 'Is_Summer', 'Quarter', 'Savings_Lag_1', 'Is_December']
    # Nos aseguramos de que existan por si acaso no generaste is_summer
    cols_validas = [c for c in columnas_externas if c in X.columns]

    # Para evitar un proceso extremadamente lento de SARIMAX por culpa de auto_arima con muchas variables,
    # apagamos estacionalidad si hay demasiadas variaciones bruscas intermensuales o histórico muy corto.
    model_sarimax = train_sarimax_model(X[cols_validas], y, seasonal=False, m=6, use_log=True)

    model_lstm, (scaler_X, scaler_y) = train_lstm_model(X, y)

    model_ensemble = train_voting_ensemble(estimators=[
        ("hgb", model_hgb),
        ("xgb", model_xgb)
        ], X=X, y=y)

    # 2. EVALUAR
    _, test_df = load_train_test_data(filename=args.dataset_name)
    test_df = preprocess_data(test_df, verbose=False)
    test_df = add_features(test_df, verbose=False)

    # Eliminar cualquier NaN residual que haya quedado en el test (de importaciones antiguas si las hubiera)
    test_df = test_df.dropna() 

    X_test, y_test = split_target(test_df)

    print("\n\n¡Datos listos para evaluar!\n\n")

    res_lr = evaluate_predictions(y_test, model_lr.predict(X_test), "Linear Regression")
    res_rf = evaluate_predictions(y_test, model_rf.predict(X_test), "Random Forest")
    res_xgb = evaluate_predictions(y_test, model_xgb.predict(X_test), "XGBoost")
    res_hgb = evaluate_predictions(y_test, model_hgb.predict(X_test), "HistGradientBoosting")
    res_arima = evaluate_predictions(y_test, evaluate_arima(y_test, model_arima), "ARIMA")
    res_sarimax = evaluate_predictions(y_test, evaluate_sarimax_wfv(X[cols_validas], y, X_test[cols_validas], y_test, model_sarimax, use_log=True), "SARIMAX")

    y_test_aligned_lstm, y_pred_lstm = evaluate_lstm(X_test, y_test, model_lstm, scaler_X, scaler_y, time_steps=3)
    res_lstm = evaluate_predictions(y_test_aligned_lstm, y_pred_lstm, "LSTM")

    res_ensemble = evaluate_predictions(y_test, model_ensemble.predict(X_test), "Ensemble")

    results = pd.DataFrame([res_lr, res_rf, res_xgb, res_hgb, res_arima, res_sarimax, res_lstm, res_ensemble])
    print(results)
    
    # Comparativa final de errores apilados
    plot_model_comparison_metrics(results, metric='MAE', title='Comparativa Evaluando Mejoras en Features + Ensamblado', save_plot=True, dataset_name=args.dataset_name)

    # Reestructuramos la comparación de predicciones
    # Diccionario con todos los nombres de los modelos evaluados como clave y la predicción como valor.
    # Excluimos LSTM momentaneamente de la gráfica conjunta porque sus indices son más cortos (y_test_aligned_lstm vs y_test)
    predicciones_modelos = {
        "Ensemble": model_ensemble.predict(X_test),
        "XGBoost": model_xgb.predict(X_test),
        "Random Forest": model_rf.predict(X_test),
        "HistGradientBoosting": model_hgb.predict(X_test),
        "Linear Regression": model_lr.predict(X_test),
        "ARIMA": evaluate_arima(y_test, model_arima),
        "SARIMAX": evaluate_sarimax_wfv(X[cols_validas], y, X_test[cols_validas], y_test, model_sarimax, use_log=True)
    }

    # Lanzamos el gráfico para todos los modelos juntos comparado contra y_test real
    plot_actual_vs_predicted_multiple(y_test, predicciones_modelos, dates=y_test.index, save_plot=True, dataset_name=args.dataset_name)

    # El LSTM lo pintamos solito (o por ejemplo contra el Ensemble que es muy bueno) ya que su dimensión es diferente
    # plot_actual_vs_predicted(y_test_aligned_lstm, y_pred_lstm, dates=y_test_aligned_lstm.index, model_name="LSTM", save_plot=True, dataset_name=args.dataset_name)

    # Opcionalmente, graficamos los histogramas de errores de algunos destacados
    # plot_error_distribution(y_test, predicciones_modelos["Ensemble"], model_name="Ensemble", save_plot=True, dataset_name=args.dataset_name)
    # plot_error_distribution(y_test_aligned_lstm, y_pred_lstm, model_name="LSTM", save_plot=True, dataset_name=args.dataset_name)

    # Solución al error de feature_importances_:
    # Un "VotingRegressor" (Ensemble) combina modelos y NO TIENE internamente el atributo global de pesos.
    # Debemos extraer las importancias de features de un modelo que SÍ los soporta como un árbol. (Ej: XGBoost o Random Forest)
    # plot_feature_importance(feature_names=X.columns, importances=model_xgb.feature_importances_, top_n=10, model_name="XGBoost", save_plot=True, dataset_name=args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar y evaluar modelos de predicción de gastos.')
    parser.add_argument('--dataset', type=str, default='db_orig', help='Nombre del dataset a usar')
    parser.add_argument('--dataset_name', type=str, default='orig', help='Nombre del dataset para guardar')
    args = parser.parse_args()
    main(args)
