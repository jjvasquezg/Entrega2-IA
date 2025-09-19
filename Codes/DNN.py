# --- Librerías ---
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
from keras import models, layers, regularizers, metrics, optimizers

# 3. PROCESAMIENTO DE DATOS

def cargar_y_preprocesar_datos(path="data.csv"):
    """Carga dataset y genera variables derivadas"""
    df = pd.read_csv(path, header=0)

    # Features derivadas
    df['Ram_GB'] = df['Ram'].str.replace('GB','', regex=False).astype(int)
    df['Weight_kg'] = df['Weight'].str.replace('kg','', regex=False).str.strip().astype(float)

    res = df['ScreenResolution'].str.extract(r'(?P<width>\d+)[xX](?P<height>\d+)')
    df['ScreenWidth'] = pd.to_numeric(res['width'], errors='coerce')
    df['ScreenHeight'] = pd.to_numeric(res['height'], errors='coerce')
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False)

    # Eliminar columnas poco útiles
    df = df.drop(columns=['Product','laptop_ID','ScreenResolution','Ram','Weight'])

    X = df.drop(columns=['Price_euros'])
    y = df['Price_euros']

    # Identificar columnas
    num_cols = X.select_dtypes(include=['int64','float64','bool']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    return X, y, num_cols, cat_cols


def crear_preprocesador(num_cols, cat_cols):
    """Define pipelines de preprocesamiento"""
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    return preprocessor


def dividir_datos(X, y):
    """Divide datos en train/val/test con proporción 70/15/15"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


# 4. ENTRENAMIENTO DE MODELOS

def evaluar_modelo(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evalúa un modelo y devuelve métricas RMSE y R2"""
    res = {}
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    res['RMSE_train'] = root_mean_squared_error(y_train, y_pred_train)
    res['R2_train'] = r2_score(y_train, y_pred_train)

    res['RMSE_val'] = root_mean_squared_error(y_val, y_pred_val)
    res['R2_val'] = r2_score(y_val, y_pred_val)

    res['RMSE_test'] = root_mean_squared_error(y_test, y_pred_test)
    res['R2_test'] = r2_score(y_test, y_pred_test)

    return res


def entrenar_modelos(X_train, X_val, X_test, y_train, y_val, y_test, preprocessor):
    """Entrena kNN, RandomForest y DNN. Devuelve métricas y modelos"""
    resultados = {}

    # --- kNN ---
    knn = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', KNeighborsRegressor(n_neighbors=5))
    ])
    knn.fit(X_train, y_train)
    resultados['kNN'] = evaluar_modelo(knn, X_train, y_train, X_val, y_val, X_test, y_test)

    # --- Random Forest ---
    rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=200, random_state=42))
    ])
    rf.fit(X_train, y_train)
    resultados['Random Forest'] = evaluar_modelo(rf, X_train, y_train, X_val, y_val, X_test, y_test)

    # --- DNN ---
    # Preprocesar manualmente
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    input_dim = X_train_proc.shape[1]

    dnn = models.Sequential([
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_dim=input_dim),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(1)
    ])

    dnn.summary()

    dnn.compile(optimizer=optimizers.Adam(learning_rate=0.1), loss='mse', metrics=[metrics.RootMeanSquaredError()])
    dnn.fit(X_train_proc, y_train, validation_data=(X_val_proc, y_val), validation_batch_size=X_val_proc.shape[0],
            epochs=32, batch_size=32, verbose=0)

    def evaluar_dnn(model, X_train, y_train, X_val, y_val, X_test, y_test):
        res = {}
        y_pred_train = model.predict(X_train).flatten()
        y_pred_val = model.predict(X_val).flatten()
        y_pred_test = model.predict(X_test).flatten()

        res['RMSE_train'] = root_mean_squared_error(y_train, y_pred_train)
        res['R2_train'] = r2_score(y_train, y_pred_train)

        res['RMSE_val'] = root_mean_squared_error(y_val, y_pred_val)
        res['R2_val'] = r2_score(y_val, y_pred_val)

        res['RMSE_test'] = root_mean_squared_error(y_test, y_pred_test)
        res['R2_test'] = r2_score(y_test, y_pred_test)

        return res

    resultados['DNN'] = evaluar_dnn(dnn, X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)

    return resultados, rf, preprocessor, dnn


# 5. PRUEBA CON MUESTRA ARTIFICIAL

def prueba_muestra_artificial(modelo, preprocessor):
    # Muestra artificial generada
    nueva_muestra = pd.DataFrame([{
        "Company": "Dell",
        "TypeName": "Gaming",
        "Inches": 15.6,
        "Cpu": "Intel Core i7 2.8GHz",
        "Ram_GB": 16,
        "Memory": "512GB SSD",
        "Gpu": "Nvidia GTX 1050",
        "OpSys": "Windows 10",
        "Weight_kg": 2.5,
        "ScreenWidth": 1920,
        "ScreenHeight": 1080,
        "Touchscreen": False
    }])

    # Si el modelo es pipeline (ej: Random Forest con preprocesador incluido)
    if isinstance(modelo, Pipeline):
        pred = modelo.predict(nueva_muestra)
    else:
        # Para DNN necesitamos preprocesar manualmente
        nueva_proc = preprocessor.transform(nueva_muestra)
        pred = modelo.predict(nueva_proc)

    print("Predicción de precio en euros:", pred[0])
    return pred[0]


# MAIN

if __name__ == "__main__":

    random_state=42

    # Cargar datos
    X, y, num_cols, cat_cols = cargar_y_preprocesar_datos("../data.csv")
    preprocessor = crear_preprocesador(num_cols, cat_cols)
    X_train, X_val, X_test, y_train, y_val, y_test = dividir_datos(X, y)

    # Entrenar modelos
    resultados, rf, preprocessor, dnn = entrenar_modelos(
        X_train, X_val, X_test, y_train, y_val, y_test, preprocessor
    )

    # Mostrar tabla comparativa
    df_resultados = pd.DataFrame(resultados).T
    print("\n=== Resultados comparativos ===")
    print(df_resultados)

    # Probar con muestra artificial
    print("\n=== Prueba con muestra artificial ===")
    prueba_muestra_artificial(rf, preprocessor)
