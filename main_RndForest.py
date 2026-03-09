import os
import logging
import warnings
import argparse
import pandas as pd

# ==========================================
# CONFIGURACIÓN DE LOGS Y WARNINGS
# ==========================================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_WARNINGS"] = "FALSE"
warnings.simplefilter(action='ignore', category=Warning)
warnings.filterwarnings('ignore')

from src.data.loader import load_raw_data, load_train_test_data, split_target
from src.data.preparation import train_val_test_split, preprocess_data
from src.features.builder import add_features
from src.models.trainer import train_sklearn_model, save_model
from src.evaluation.metrics import evaluate_predictions
from src.visualization.plots import plot_actual_vs_predicted, plot_feature_importance

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


    # 7. ENTRENAR (Baseline y Avanzado)
    model_rf = train_sklearn_model(X, y, model_type="rf")

    # 8. EVALUAR
    _, test_df = load_train_test_data(filename=args.dataset_name)
    test_df = preprocess_data(test_df, verbose=False)
    test_df = add_features(test_df, verbose=False)

    X_test, y_test = split_target(test_df)

    print("\n\n¡Datos listos para evaluar!\n\n")

    y_pred = model_rf.predict(X_test)
    res_rf = evaluate_predictions(y_test, model_rf.predict(X_test), "Random Forest")

    results = pd.DataFrame([res_rf])
    print(results)
    
    # 9. Visualización
    plot_actual_vs_predicted(y_test, y_pred, dates=y_test.index, 
                             model_name="RandomForest", save_plot=True, 
                             dataset_name=args.dataset_name)

    # 10. Guardar el modelo
    save_model(model_rf, f"best_model_rndForest_{args.dataset_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entrenar y evaluar modelos de predicción de gastos.')
    parser.add_argument('--dataset', type=str, default='db_orig', help='Nombre del dataset a usar')
    parser.add_argument('--dataset_name', type=str, default='orig', help='Nombre del dataset para guardar')
    args = parser.parse_args()
    main(args)
