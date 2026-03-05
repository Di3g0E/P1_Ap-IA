# P1_Ap-IA: Gestión y Predicción de Finanzas Personales

Este repositorio contiene la primera práctica de la asignatura **Aprendizaje Automático (Ap-IA)**. El objetivo es analizar y predecir el comportamiento financiero personal utilizando técnicas de Machine Learning.

## 🚀 Instrucciones de Ejecución

> Para una guía completa de configuración desde cero (incluida la instalación de Python y uv), consulta [SETUP.md](SETUP.md).

### Inicio Rápido

Desde la carpeta `practicas/`:

```powershell
# Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# Instalar dependencias (solo la primera vez)
uv pip install -r requirements.txt --link-mode copy
```

### Ejecutar los Notebooks

1. Abre VS Code en `P1_Ap-IA/`.
2. Abre cualquier notebook de `notebooks/`.
3. Selecciona el kernel `.venv (Python 3.13.0)`.
4. Ejecuta las celdas.

## 📁 Estructura del Proyecto

- `data/`: Gestión de datos.
    - `raw/`: Datos originales (`db.csv`).
    - `processed/`: Datos procesados listos para modelos.
- `doc/`: Documentación técnica y memoria (LaTeX).
- `models/`: Binarios de modelos entrenados.
- `notebooks/`: Experimentos y análisis.
    - `01_test_carga.ipynb`: Validación de carga de datos.
- `papers/`: Referencias científicas.
- `src/`: Código fuente modular.
    - `data/`: Scripts de carga y limpieza.
    - `features/`: Ingeniería de variables.
    - `models/`: Entrenamiento y arquitectura de modelos.
    - `evaluation/`: Métricas y validación.
    - `utils/`: Utilidades transversales.
- `README.md`: Instrucciones y descripción del proyecto.
- `SETUP.md`: Guía completa de configuración del entorno.

## 📚 Referencias (Estado del Arte)

Se han utilizado los siguientes artículos para el desarrollo de la solución:
- *Personal Finance Management and Prediction using ML Algorithms*
- *Personal Finance Tracker with Spending Behavior Analysis*
- *Portrait of an Online Shopper*
- *WONGA: The Future of Personal Finance*

---
**Autores:** [Diego - Sofía]