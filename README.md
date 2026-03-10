# P1_Ap-IA: Gestión y Predicción de Finanzas Personales

Este repositorio contiene la primera práctica de la asignatura **Aprendizaje Automático (Ap-IA)**. El objetivo es analizar y predecir el comportamiento financiero personal utilizando técnicas de Machine Learning.

## Instrucciones de Ejecución

> Para una guía completa de configuración desde cero (incluida la instalación de Python y uv), consulta [SETUP.md](SETUP.md).

### Inicio Rápido

### Entorno Virtual Compartido (Importante)

Desde tu terminal (ubicada en la raíz `P1_Ap-IA/`):

**Si usas Windows (Powershell):**
```powershell
.venv\Scripts\Activate.ps1
```

**Si no existe el entorno todavía**, lo debes crear de la siguiente forma e instalar las librerías propias de la práctica 1:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
uv pip install -r P1_Ap-IA/requirements.txt --link-mode copy
```

### Ejecutar el Pipeline Principal

El archivo `main.py` permite entrenar o cargar los modelos que mejor se han comportado en las pruebas realizadas desde la línea de comandos.

**Opciones disponibles:**
*   `--model`: Modelo a utilizar (`"Random Forest"`, `"HistGradBoosting"`, `"ARIMA"`).
*   `--train`: (Opcional) Indica que el modelo debe entrenarse desde cero.
*   `--plot`: (Opcional) Genera y guarda visualizaciones de resultados.
*   `--dataset`: Nombre del archivo CSV en `data/raw/` (por defecto `db_orig`).
*   `--dataset_name`: Etiqueta para los archivos de salida (por defecto `orig`).

**Ejemplos de uso:**

1. **Cargar el entorno**
```powershell
.venv\Scripts\Activate.ps1
````

2. **Entrenar y evaluar un modelo** (por ejemplo, Random Forest):
```powershell
python main.py --model "Random Forest" --dataset db_trunc_8779 --dataset_name trunc_8779 --train
```

3. **Cargar y evaluar un modelo ya entrenado** (con gráficas):
```powershell
python main.py --model "Random Forest" --dataset db_trunc_8779 --dataset_name trunc_8779 --plot
```

## 📁 Estructura del Proyecto

Este proyecto sigue la siguiente estructura:

- `data/`: Gestión de datos.
    - `raw/`: Datos originales e inmutables.
    - `processed/`: Datos limpios y listos para entrenamiento.
- `doc/`: Documentación técnica y memoria (LaTeX).
- `models/`: Archivos binarios de modelos entrenados (`.pkl`, `.h5`).
- `playground/`: Espacio para pruebas rápidas y cuadernos (`.ipynb`).
- `references/`: Artículos científicos y papers de referencia.
- `src/`: Código fuente modular.
    - `data/`: Scripts de carga y limpieza.
    - `features/`: Ingeniería de variables (Feature Engineering).
    - `models/`: Arquitectura y entrenamiento de modelos.
    - `evaluation/`: Métricas y reportes de rendimiento.
    - `utils/`: Utilidades transversales (logging, helpers).
    - `experiments/`: Experimentos y análisis de resultados.
    - `visualization/`: Funciones de visualización de resultados.
- `main.py`: Punto de entrada principal para ejecución.
- `requirements.txt`: Dependencias del proyecto con versiones fijas.
- `SETUP.md`: Guía de configuración del entorno.

## 📚 Referencias (Estado del Arte)

Se han utilizado los siguientes artículos para el desarrollo de la solución:
- *Personal Finance Management and Prediction using ML Algorithms*
- *Personal Finance Tracker with Spending Behavior Analysis*
- *Portrait of an Online Shopper*
- *WONGA: The Future of Personal Finance*

---
**Autores:** [Diego - Sofía]