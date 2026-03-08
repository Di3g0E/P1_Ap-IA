import os
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_matrix(df, target="Expenses", title="Correlation Matrix", save=False, view=True, verbose=False):
    """
    Visualiza la matriz de correlación y agrupa las features por su nivel de 
    correlación con la variable 'target'.
    """
    # 1. Preparar datos
    df_num = df.select_dtypes(include=['number'])
    if target not in df_num.columns:
        print(f"Warning: '{target}' no encontrado en las columnas numéricas. Usando correlación general.")
        target_corr = None
    else:
        target_corr = df_num.corr()[target].drop(target)

    corr = df_num.corr()

    # 2. Configurar el gráfico
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("coolwarm", 20)
    sns.heatmap(corr, annot=True, cmap=palette, fmt=".2f", vmin=-1, vmax=1, center=0,
                cbar_kws={'label': 'Grado de correlación (bins de 0.1)'})
    plt.title(title, fontsize=15, pad=20)

    # 3. Lógica de agrupación (0-10, 10-20...)
    groups = {}
    if target_corr is not None:
        for i in range(0, 100, 10):
            label = f"{i}-{i+10}%"
            mask = (target_corr.abs() * 100 >= i) & (target_corr.abs() * 100 < i+10)
            feats = target_corr[mask].index.tolist()
            if feats:
                groups[label] = [f"{f} ({target_corr[f]:.2f})" for f in feats]

    # 4. Lógica de guardado
    if save:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(base_path, "reports", "plots")
        os.makedirs(save_dir, exist_ok=True)

        filename = save if isinstance(save, str) else "correlation_matrix.png"
        if not filename.endswith(('.png', '.jpg', '.pdf')): filename += ".png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if verbose:
        if save:
            print(f"\nGráfica guardada en: {save_path}")

        print(f"\nNivel de correlación respecto a '{target}':")
        for label in sorted(groups.keys(), key=lambda x: int(x.split('-')[0]), reverse=True):
            print(f"  {label.ljust(8)}: {', '.join(groups[label])}")
        return groups

    # 5. Lógica de visualización y retorno
    if view:
        plt.show()
    else:
        plt.close()
        return None

def plot_model_comparison(results_df):
    # Lógica para comparar RMSE, MAE, etc.
    pass
