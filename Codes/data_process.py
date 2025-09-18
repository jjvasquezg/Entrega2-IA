# --- Librerías ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# --- Cargar dataset ---
df = pd.read_csv("data.csv")

# --- Preprocesamiento inicial manual ---
# Variables derivadas
df['Ram_GB'] = df['Ram'].str.replace('GB','', regex=False).astype(int)
df['Weight_kg'] = df['Weight'].str.replace('kg','', regex=False).str.strip().astype(float)

res = df['ScreenResolution'].str.extract(r'(?P<width>\d+)[xX](?P<height>\d+)')
df['ScreenWidth'] = pd.to_numeric(res['width'], errors='coerce')
df['ScreenHeight'] = pd.to_numeric(res['height'], errors='coerce')
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False)

# Eliminamos columnas poco útiles o con cardinalidad muy alta (ej: Product, laptop_ID)
df = df.drop(columns=['Product','laptop_ID','ScreenResolution','Ram','Weight'])

# --- Definir variables predictoras y target ---
X = df.drop(columns=['Price_euros'])
y = df['Price_euros']

# Identificar columnas categóricas y numéricas
num_cols = X.select_dtypes(include=['int64','float64','bool']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print("Numéricas:", num_cols)
print("Categóricas:", cat_cols)

# --- Pipelines de preprocesamiento ---

# Para numéricas: imputar nulos con media + estandarizar
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Para categóricas: imputar nulos con "missing" + OneHotEncoder
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combinamos transformadores
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Opcional: añadir reducción de dimensionalidad (ejemplo PCA)
# Aquí reducimos a 20 componentes tras codificación
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('dim_reduction', PCA(n_components=20))
])

# --- División de datos ---
# Primero train/test (85/15), luego train/val (70/15/15 total)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)
# (0.176 ≈ 0.15 / 0.85 para mantener 70/15/15)

print("Tamaños finales:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# --- Ajustar pipeline (solo preprocesamiento por ahora) ---
pipeline.fit(X_train)

# Transformar datasets
X_train_proc = pipeline.transform(X_train)
X_val_proc = pipeline.transform(X_val)
X_test_proc = pipeline.transform(X_test)

print("Shape después de pipeline:", X_train_proc.shape)
