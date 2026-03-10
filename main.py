import os
import warnings
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings('ignore')

from src.data.loader import load_raw_data, load_train_test_data, split_target
from src.data.preparation import train_val_test_split, preprocess_data
from src.features.builder import add_features
from src.models.trainer import train_sklearn_model, train_arima_model, save_model, load_model
from src.evaluation.metrics import evaluate_predictions, evaluate_arima
from src.visualization.plots import (
    plot_actual_vs_predicted,
    plot_error_distribution
)


def main(args):
    model_mapping = {
        "Random Forest": "rf",
        "HistGradBoosting": "hgb",
        "ARIMA": "arima"
    }
    
    selected_model_name = args.model
    if selected_model_name not in model_mapping:
        print(f"Error: Modelo '{selected_model_name}' no soportado. Elija entre: {list(model_mapping.keys())}")
        return

    print(f"\n--- Iniciando Pipeline Final para el modelo: {selected_model_name} ---\n")

    print("1. Carga y preprocesado de datos...")
    df_raw = load_raw_data(args.dataset)
    
    train_val_test_split(df_raw, dataset_name=args.dataset_name, test_size=0.2, val_size=0, verbose=False)
    
    train_df, _ = load_train_test_data(filename=args.dataset_name)
    train_df = preprocess_data(train_df, verbose=False)
    train_df = add_features(train_df, verbose=False)

    model_type = model_mapping[selected_model_name]
    model_filename = f"{selected_model_name.replace(' ', '_').lower()}_{args.dataset_name}"
    
    if args.train:
        print(f"2. Entrenando {selected_model_name}...")
        X_train, y_train = split_target(train_df)
        
        if model_type in ["rf", "hgb"]:
            model = train_sklearn_model(X_train, y_train, model_type=model_type)
        elif model_type == "arima":
            model = train_arima_model(y_train)
        
        # Guardar modelo
        save_model(model, model_filename)
    else:
        print(f"2. Cargando modelo {selected_model_name}...")
        try:
            model = load_model(model_filename)
        except Exception as e:
            print(f"\n[!] Error: No se pudo encontrar o cargar el modelo '{model_filename}' en la carpeta models.")
            print("Asegúrate de haberlo entrenado previamente ejecutando el script con la bandera '--train'.")
            return

    # Evaluación
    print("3. Evaluación del modelo...")
    _, test_df = load_train_test_data(filename=args.dataset_name)
    test_df = preprocess_data(test_df, verbose=False)
    test_df = add_features(test_df, verbose=False)
    test_df = test_df.dropna()
    
    X_test, y_test = split_target(test_df)

    if model_type in ["rf", "hgb"]:
        y_pred = model.predict(X_test)
        results = evaluate_predictions(y_test, y_pred, selected_model_name)
    elif model_type == "arima":
        y_pred = evaluate_arima(y_test, model)
        results = evaluate_predictions(y_test, y_pred, selected_model_name)

    print("\nResultados obtenidos:")
    for metric, value in results.items():
        if metric != "Modelo":
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    # Visualización
    if args.plot:
        print("\n4. Generando visualizaciones...")
        plot_actual_vs_predicted(y_test, y_pred, dates=y_test.index, model_name=selected_model_name, save_plot=True, dataset_name=args.dataset_name)
        plot_error_distribution(y_test, y_pred, model_name=selected_model_name, save_plot=True, dataset_name=args.dataset_name)

    print("\n--- Pipeline Finalizado con éxito ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline final para entrenamiento/evaluación de un modelo específico.')
    parser.add_argument('--model', type=str, required=True, 
                        choices=["Random Forest", "HistGradBoosting", "ARIMA"],
                        help='Modelo a utilizar: "Random Forest", "HistGradBoosting" o "ARIMA"')
    parser.add_argument('--train', action='store_true', help='Si se indica, el modelo será entrenado. De lo contrario, se intentará cargar uno ya existente.')
    parser.add_argument('--dataset', type=str, default='db_orig', help='Nombre del archivo CSV en data/raw (sin extensión)')
    parser.add_argument('--dataset_name', type=str, default='orig', help='Etiqueta para identificar el dataset en los archivos guardados')
    parser.add_argument('--plot', action='store_true', help='Generar y guardar gráficos de resultados')
    
    args = parser.parse_args()
    main(args)
