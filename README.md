# CastVision — Inspección automática de piezas fundidas con Visión Artificial Clásica

Proyecto de la **Entrega 3 · Visión Artificial · ITM** (docente: Cristian Mateo Hernández González).
Autor: **Sebastián Gallego**.

CastVision clasifica fotos de piezas metálicas fundidas como **OK** o **Defectuosa** usando únicamente el pipeline clásico de Visión Artificial

```
Imagen → Descriptores (HOG · LBP · GLCM) → Z-score → Clasificador (k-NN · SVM · Random Forest)
```

## Resultado principal

Se compararon 3 clasificadores × 4 descriptores (HOG, LBP, GLCM y fusión) sobre 2000 imágenes (1000 por clase, 70 % entrenamiento / 30 % validación).
Las métricas se miden sobre la clase **defectuosa**.

| Descriptor | k-NN (k=5) | SVM (RBF) | Random Forest |
|---|---|---|---|
| HOG (8100 características) | 0.557 | 0.962 | 0.877 |
| LBP (10) | 0.762 | 0.781 | 0.782 |
| **GLCM (48)** | 0.928 | 0.926 | **0.971** |
| Fusión HOG+LBP+GLCM (8158) | 0.584 | 0.962 | 0.899 |

*F1-Score de la clase defectuosa en el conjunto de validación (un solo reparto, `random_state=42`).*

**Mejor modelo: GLCM + Random Forest** → F1 = 0.971, Recall = 95.3 %, Precisión = 99.0 %, con solo 48 características y menos de 1 segundo de entrenamiento.

**¿Por qué GLCM y no HOG?** El defecto de una pieza fundida (grieta, poro) es un cambio de *textura de superficie*, no de silueta. GLCM describe textura; HOG describe formas.

> El notebook incluye además matriz de confusión y validación cruzada de 5 particiones para confirmar que el resultado no depende de un único reparto de datos.

## Estructura del repositorio

```
.
├── CastVision_entrega3.ipynb   # Notebook completo (carga, descriptores, benchmark, modelo, predicción)
├── modelo_castvision.joblib    # Modelo entrenado (Z-score + Random Forest sobre GLCM)
├── requirements.txt            # Librerías necesarias
├── resultados/                 # Tabla y gráficos generados por el notebook
└── README.md
```

El dataset **no** está incluido (pesa unos 170 MB). Descárgalo aquí y descomprímelo junto al notebook, de modo que quede `casting_data/casting_data/train/...`:
<https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product>

## Cómo ejecutarlo

```bash
# 1. Instalar las librerías (se recomienda un entorno de Anaconda o venv)
pip install -r requirements.txt

# 2. Abrir el notebook
jupyter lab CastVision_entrega3.ipynb
```

Luego: *Kernel → Restart & Run All*. El benchmark completo tarda unos pocos minutos (la parte más lenta es SVM con HOG).

Si tu dataset está en otra carpeta, cambia `RUTA_DATASET` en la celda de configuración.

## Probar una pieza nueva

Con el modelo ya entrenado (`modelo_castvision.joblib`), la última sección del notebook ofrece:

```python
predecir_pieza("ruta/a/mi_pieza.jpg")
# {'resultado': 'DEFECTUOSA', 'confianza': 0.94}
```

y un botón **Subir foto** para clasificar una imagen sin escribir código.

## Limitaciones (honestidad técnica)

- El dataset de Kaggle está balanceado (50 % defectuosas / 50 % OK). En una planta real los defectos son pocos, por lo que habría que recalibrar el umbral y volver a medir las falsas alarmas.
- Se usaron 1000 imágenes por clase; usar el dataset completo y datos de la propia planta podría mejorar la generalización.
- Las **cifras de ahorro en COP** del notebook y la presentación son **estimaciones ilustrativas** con supuestos editables, no datos reales de una empresa.
- Las imágenes son de un solo tipo de pieza, con iluminación y encuadre controlados.

## Tecnologías

Python · OpenCV · scikit-image · scikit-learn · pandas · matplotlib · Jupyter
