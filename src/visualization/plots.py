import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def plot_model_comparison_metrics(df_results, metric='MAE', title=None, save_plot=False, dataset_name=None):
    """
    Genera un gráfico de barras horizontal comparando una métrica (ej. MAE, RMSE)
    entre los diferentes modelos. Destaca el mejor modelo (el de menor error).
    """
    # Ordenar los resultados para que el mejor quede abajo
    df_sorted = df_results.sort_values(by=metric, ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(12, max(6, len(df_sorted) * 0.8)))
    sns.set_theme(style="whitegrid")
    
    # Identificar el índice del mejor modelo (mínimo error)
    best_idx = df_sorted[metric].idxmin()
    
    # Crear paleta de colores: gris para todos, y un color destacado (verde oscuro) para el mejor
    colores = ['#bdc3c7' if i != best_idx else '#2ecc71' for i in range(len(df_sorted))]
    
    # Asumimos que la columna con el nombre del modelo se llama 'Model' o 'Modelo'
    model_col = 'Model' if 'Model' in df_sorted.columns else 'Modelo'
    
    grafico = sns.barplot(x=metric, y=model_col, data=df_sorted, palette=colores, edgecolor='black')
    
    if title is None:
        title = f'Comparativa de Rendimiento de Modelos ({metric})'
        
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel(f'Euros de Error ({metric}) - ¡Menor es Mejor!', fontsize=12)
    plt.ylabel('')
    
    # Añadir el texto de los valores a las barras
    max_val = df_sorted[metric].max()
    for index, value in enumerate(df_sorted[metric]):
        # Ajustar la posición y color del texto en función del valor
        text_color = 'white' if index == best_idx else 'black'
        offset = max_val * 0.02
        if pd.isna(value):
            continue
        grafico.text(value - offset, index, f"{value:.1f} €", color=text_color, 
                     ha="right", va="center", fontweight='bold', fontsize=12)
                     
    plt.tight_layout()
    
    if save_plot:
        _save_plot("model_comparison_metrics", metric, dataset_name)
        
    plt.show()

def plot_actual_vs_predicted(y_true, y_pred, dates=None, model_name="Modelo", save_plot=False, dataset_name=None):
    """
    Grafica la serie temporal de valores reales frente a las predicciones 
    de una manera muy visual, ideal para entender visualmente el acierto del modelo.
    """
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    if dates is None:
        x_axis = range(len(y_true))
    else:
        x_axis = dates
        
    # Línea de la realidad
    plt.plot(x_axis, y_true, label='Gasto Real Registrado', color='#2c3e50', linewidth=2.5, marker='o', markersize=6)
    
    # Línea de la predicción
    plt.plot(x_axis, y_pred, label=f'Lo que predijo la IA ({model_name})', color='#e74c3c', linewidth=2.5, linestyle='--', marker='s', markersize=6)
    
    # Rellenar la diferencia visualmente
    plt.fill_between(x_axis, y_true, y_pred, color='#e74c3c', alpha=0.1, label='Margen de Error')
    
    plt.title(f'Predicción vs Realidad: ¿Cómo de bien lo hizo {model_name}?', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Fecha o Periodo', fontsize=12)
    plt.ylabel('Monto (€)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    
    if dates is not None:
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    
    if save_plot:
        _save_plot(model_name, 'actual_vs_predicted', dataset_name)
        
    plt.show()


def plot_actual_vs_predicted_multiple(y_true, dict_y_preds, dates=None, save_plot=False, dataset_name=None):
    """
    Grafica la serie temporal de valores reales frente a las predicciones 
    de MÚLTIPLES modelos a la vez. Pensado para ver la varianza general en una sola gráfica.
    dict_y_preds: Diccionario con clave (Nombre Modelo) y valor (lista o serie de predicción).
    """
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    if dates is None:
        x_axis = range(len(y_true))
    else:
        x_axis = dates
        
    # Línea de la realidad (MÁS GRUESA Y RESALTADA)
    plt.plot(x_axis, y_true, label='Gasto Real Registrado', color='black', linewidth=4, marker='o', markersize=8, zorder=10)
    
    # Genera una paleta de colores para iterar entre ellos
    colores_modelos = sns.color_palette("husl", len(dict_y_preds))
    marcas = ['s', '^', 'D', 'v', 'p', '*', 'X']
    
    # Línea de la predicción por cada modelo
    for i, (model_name, y_pred) in enumerate(dict_y_preds.items()):
        plt.plot(x_axis, y_pred, label=model_name, 
                 color=colores_modelos[i], linewidth=2, linestyle='--', marker=marcas[i%len(marcas)], markersize=5, alpha=0.8)
    
    plt.title('Comparativa General: Predicciones Modelos vs Gasto Real', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Fecha o Periodo', fontsize=12)
    plt.ylabel('Monto (€)', fontsize=12)
    
    # Ubicamos la leyenda fuera de la gráfica para no tapar los dibujos si hay muchos.
    plt.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
    
    if dates is not None:
        plt.xticks(rotation=45)
        
    plt.tight_layout()
    
    if save_plot:
        _save_plot('Multi_Modelos', 'actual_vs_predicted_all', dataset_name)

    plt.show()

def plot_error_distribution(y_true, y_pred, model_name="Modelo", save_plot=False, dataset_name=None):
    """
    Visualiza la distribución de los errores (diferencia entre real y predicho).
    Ayuda a entender si el modelo tiende a cobrar de más (sobreestimar) o de menos (subestimar).
    """
    errores = np.array(y_true) - np.array(y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.histplot(errores, kde=True, color='#3498db', edgecolor='black', bins=15)
    
    # Líneas de referencia
    plt.axvline(x=0, color='#2ecc71', linestyle='--', linewidth=2.5, label='Error 0 (Perfecto)')
    
    promedio_error = np.mean(errores)
    color_promedio = '#f39c12' if abs(promedio_error) > 10 else '#27ae60'
    plt.axvline(x=promedio_error, color=color_promedio, linestyle='-', linewidth=2, label=f'Desviación Media: {promedio_error:.1f} €')
    
    plt.title(f'Análisis de la Desviación: ¿Qué tipo de errores comete {model_name}?', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Diferencia (Realidad - Predicción) en €\n<-- Predijo de MÁS   |   Predijo de MENOS -->', fontsize=12)
    plt.ylabel('Frecuencia (Nº de veces)', fontsize=12)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if save_plot:
        _save_plot(model_name, 'error_distribution', dataset_name)

    plt.show()

def plot_feature_importance(feature_names, importances, top_n=10, model_name="Modelo", save_plot=False, dataset_name=None):
    """
    Muestra los factores o características más importantes en los que se fija el modelo.
    """
    df_imp = pd.DataFrame({
        'Factor': feature_names,
        'Importancia': importances
    }).sort_values(by='Importancia', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    grafico = sns.barplot(x='Importancia', y='Factor', data=df_imp, palette='magma', edgecolor='black')
    
    plt.title(f'Top {top_n} Factores Clave para la Predicción ({model_name})', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Peso / Importancia Relativa', fontsize=12)
    plt.ylabel('')
    
    plt.tight_layout()
    
    if save_plot:
        _save_plot(model_name, 'feature_importance', dataset_name)

    plt.show()


def _save_plot(model_name, metric, dataset_name=None, save_path="P1_Ap-IA/reports/plots"):
    os.makedirs(save_path, exist_ok=True)
    if dataset_name:
        filename = f"{dataset_name}_{model_name}_{metric}.png"
    else:
        filename = f"{model_name}_{metric}.png"
    
    plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
