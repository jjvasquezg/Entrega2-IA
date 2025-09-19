
# Análisis Predictivo de Precios de Portátiles

Este proyecto forma parte del laboratorio de Machine Learning y Deep Learning de la Universidad EAFIT. Su objetivo principal es aplicar técnicas fundamentales de análisis de datos y modelado supervisado para predecir el precio de computadores portátiles, partiendo de un conjunto de características técnicas.

Se ha elegido un conjunto de datos público disponible en Kaggle, que incluye especificaciones como procesador, memoria, resolución de pantalla, sistema operativo, entre otras. A partir de estos datos, se desarrolla un flujo completo que incluye análisis exploratorio, limpieza, transformación, entrenamiento y evaluación de modelos.

Este repositorio contiene el código, gráficos, resultados y documentación técnica del proyecto. Para un análisis detallado y explicaciones técnicas, consulte la Wiki de este repositorio.

---

## Integrantes

- Juan Jose Vasquez Gomez
- Santiago Alvarez Peña

---

## Tecnologías Utilizadas en el Proyecto

Este proyecto combina diferentes herramientas y librerías de **Python** para llevar a cabo un flujo completo de análisis, preprocesamiento, modelado y evaluación de datos.

---

### 📊 Manipulación y Análisis de Datos
- **Pandas** → Manejo de DataFrames, limpieza de datos y transformaciones.
- **NumPy** → Operaciones numéricas eficientes y soporte para arrays multidimensionales.

---

### 📈 Visualización
- **Matplotlib** → Creación de gráficos básicos y personalizados.
- **Seaborn** → Visualizaciones estadísticas más avanzadas (distribuciones, correlaciones, boxplots, heatmaps).

---

### 🤖 Machine Learning Clásico
- **Scikit-learn** → Implementación de pipelines, preprocesamiento y modelos como:
  - **k-Nearest Neighbors (kNN)**
  - **Random Forest Regressor**
- Herramientas de validación y métricas de desempeño.

---

### 🧠 Deep Learning
- **TensorFlow / Keras** → Construcción y entrenamiento de redes neuronales profundas (DNN).
  - Capas densas con funciones de activación ReLU.
  - Regularización con **Dropout** y **L2**.
  - Optimizador **Adam** con ajuste de tasa de aprendizaje.
  - Callbacks como **EarlyStopping** y **ReduceLROnPlateau** para mejorar la generalización.

---

### ⚙️ Infraestructura y Organización
- **Pipelines de Scikit-learn** → Integración de preprocesamiento (escalado, codificación de categóricas) con modelos predictivos.
- **Markdown / Wiki de GitHub** → Documentación clara del flujo de trabajo, análisis exploratorio y resultados.

---
