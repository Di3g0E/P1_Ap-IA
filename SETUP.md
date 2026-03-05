# Guía de Configuración del Entorno — P1_Ap-IA

Esta guía explica cómo configurar desde cero el entorno de desarrollo necesario para ejecutar los notebooks y scripts de esta práctica.

## 1. Requisitos previos

### Python 3.13+
Descarga e instala Python desde [python.org](https://www.python.org/downloads/). Durante la instalación, asegúrate de marcar **"Add Python to PATH"**.

Verifica la instalación:
```powershell
python --version
```

### uv (Gestor de paquetes)
[uv](https://docs.astral.sh/uv/) es un gestor de paquetes ultrarrápido para Python. Para instalarlo:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verifica la instalación:
```powershell
uv --version
```

## 2. Crear el entorno virtual

El entorno virtual es **compartido** entre todas las prácticas y se ubica en la carpeta `practicas/.venv`. Desde la carpeta `practicas/`:

```powershell
uv venv .venv
```

## 3. Activar el entorno virtual

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
source .venv/bin/activate
```

## 4. Instalar dependencias

Desde la carpeta `practicas/` (donde está el `requirements.txt`):

```powershell
uv pip install -r requirements.txt --link-mode copy
```

## 5. Verificar la instalación

Ejecuta el notebook de test para comprobar que todo funciona:

1. Abre VS Code en la carpeta `P1_Ap-IA/`.
2. Abre `notebooks/01_test_carga.ipynb`.
3. Selecciona como kernel el intérprete de `.venv` (aparecerá como `.venv (Python 3.13.0)`).
4. Ejecuta todas las celdas. Deberías ver el dataset cargado correctamente.

## 6. Dependencias instaladas

| Paquete        | Versión  | Uso                                      |
|----------------|----------|------------------------------------------|
| `ipykernel`    | 7.2.0    | Ejecutar notebooks en VS Code            |
| `pandas`       | 3.0.1    | Manipulación y análisis de datos          |
| `numpy`        | 2.4.2    | Cálculos numéricos                       |
| `matplotlib`   | 3.10.8   | Visualización de datos                   |
| `seaborn`      | 0.13.2   | Gráficos estadísticos                    |
| `scikit-learn`  | 1.8.0    | Modelos de Machine Learning              |

## 7. Creación del .gitignore

Crea un archivo .gitignore en la carpeta practicas/P1_Ap-IA/ con el siguiente contenido:

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/

# Modelos (archivos pesados)
models/*.pkl
models/*.h5
models/*.joblib

# Ignorar carpetas de habilidades
.agent/

# Entorno
.env

# IDEs
.vscode/
.idea/

# Jupyter Notebooks
.ipynb_checkpoints/

# Archivos de configuración
.gitignore
```