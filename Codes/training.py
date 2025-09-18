# --- Librerías ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

import tensorflow as tf
from tensorflow.keras import models, layers, regularizers

# --- Cargar dataset ---
df = pd.read_csv("data.csv")

# --- Limpieza inicial (similar al punto anterior) ---
df['Ram_GB'] = df['Ram'].str.replace('GB','', regex=False).astype(int)
df['Weight_kg'] = df['Weight'].str.replace('kg','', regex=False).str.strip().astype(float)

res = df['ScreenResolution'].str.extract(r'(?P<width>\d+)[xX](?P<height>\d+)')
df['ScreenWidth'] = pd.to_numeric(res['width'], errors='coerce')
df['ScreenHeight'] = pd.to_numeric(res['height'], errors='coerce')
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False)

df = df.drop(columns=['Product','laptop_ID','ScreenResolution','Ram','Weight'])

X = df.drop(columns=['Price_euros'])
y = df['Price_euros']

num_cols = X.select_dtypes(include=['int64','float64','bool']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# --- Transformadores ---
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

# --- División Train/Val/Test (70/15/15) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)

# --- Función de evaluación ---
def evaluar_modelo(model, X_train, y_train, X_val, y_val, X_test, y_test):
    resultados = {}
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    resultados['RMSE_train'] = mean_squared_error(y_train, y_pred_train, squared=False)
    resultados['R2_train'] = r2_score(y_train, y_pred_train)
    
    resultados['RMSE_val'] = mean_squared_error(y_val, y_pred_val, squared=False)
    resultados['R2_val'] = r2_score(y_val, y_pred_val)
    
    resultados['RMSE_test'] = mean_squared_error(y_test, y_pred_test, squared=False)
    resultados['R2_test'] = r2_score(y_test, y_pred_test)
    
    return resultados

# --- 4.a. Modelos ---
# 1) kNN
knn = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor(n_neighbors=5))
])
knn.fit(X_train, y_train)
res_knn = evaluar_modelo(knn, X_train, y_train, X_val, y_val, X_test, y_test)

# 2) Random Forest
rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=42))
])
rf.fit(X_train, y_train)
res_rf = evaluar_modelo(rf, X_train, y_train, X_val, y_val, X_test, y_test)

# 3) Deep Neural Network (TensorFlow)
# Preprocesamiento fuera del pipeline porque Keras requiere arrays
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

dnn.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

dnn.fit(X_train_proc, y_train, validation_data=(X_val_proc, y_val), epochs=30, batch_size=32, verbose=0)

# Evaluación DNN
def evaluar_dnn(model, X_train, y_train, X_val, y_val, X_test, y_test):
    resultados = {}
    y_pred_train = model.predict(X_train).flatten()
    y_pred_val = model.predict(X_val).flatten()
    y_pred_test = model.predict(X_test).flatten()
    
    resultados['RMSE_train'] = mean_squared_error(y_train, y_pred_train, squared=False)
    resultados['R2_train'] = r2_score(y_train, y_pred_train)
    
    resultados['RMSE_val'] = mean_squared_error(y_val, y_pred_val, squared=False)
    resultados['R2_val'] = r2_score(y_val, y_pred_val)
    
    resultados['RMSE_test'] = mean_squared_error(y_test, y_pred_test, squared=False)
    resultados['R2_test'] = r2_score(y_test, y_pred_test)
    
    return resultados

res_dnn = evaluar_dnn(dnn, X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)

# --- Tabla comparativa ---
resultados = pd.DataFrame({
    "Modelo": ["kNN", "Random Forest", "Deep Neural Net"],
    "RMSE_train": [res_knn['RMSE_train'], res_rf['RMSE_train'], res_dnn['RMSE_train']],
    "R2_train": [res_knn['R2_train'], res_rf['R2_train'], res_dnn['R2_train']],
    "RMSE_val": [res_knn['RMSE_val'], res_rf['RMSE_val'], res_dnn['RMSE_val']],
    "R2_val": [res_knn['R2_val'], res_rf['R2_val'], res_dnn['R2_val']],
    "RMSE_test": [res_knn['RMSE_test'], res_rf['RMSE_test'], res_dnn['RMSE_test']],
    "R2_test": [res_knn['R2_test'], res_rf['R2_test'], res_dnn['R2_test']],
})

print(resultados)
