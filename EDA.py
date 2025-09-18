# --- Librerías ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cargar dataset ---
df = pd.read_csv("data.csv")

# --- Limpieza inicial de variables ---
df['Ram_GB'] = df['Ram'].str.replace('GB','', regex=False).astype(int)
df['Weight_kg'] = df['Weight'].str.replace('kg','', regex=False).str.strip().astype(float)

# Extraer resolución de pantalla
res = df['ScreenResolution'].str.extract(r'(?P<width>\d+)[xX](?P<height>\d+)')
df['ScreenWidth'] = pd.to_numeric(res['width'], errors='coerce')
df['ScreenHeight'] = pd.to_numeric(res['height'], errors='coerce')
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False, na=False)

# --- 2.1 Distribuciones de las variables numéricas ---
df[['Inches','Ram_GB','Weight_kg','ScreenWidth','ScreenHeight','Price_euros']].hist(bins=30, figsize=(12,8))
plt.suptitle("Distribuciones de variables numéricas")
plt.show()

# --- 2.2 Estadísticos descriptivos ---
print(df[['Inches','Ram_GB','Weight_kg','ScreenWidth','ScreenHeight','Price_euros']].describe())

# --- 2.3 Distribución de variables categóricas principales ---
for col in ['Company','TypeName','OpSys']:
    plt.figure(figsize=(10,4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Distribución de {col}")
    plt.show()

# --- 2.4 Correlaciones entre variables numéricas ---
plt.figure(figsize=(8,6))
corr = df[['Inches','Ram_GB','Weight_kg','ScreenWidth','ScreenHeight','Price_euros']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de correlación")
plt.show()

# --- 2.5 Relación entre predictoras y target ---
# Precio vs RAM
plt.figure(figsize=(6,4))
sns.boxplot(x='Ram_GB', y='Price_euros', data=df)
plt.title("Precio vs RAM (GB)")
plt.show()

# Precio vs Resolución de pantalla
plt.figure(figsize=(6,4))
sns.scatterplot(x='ScreenWidth', y='Price_euros', data=df, alpha=0.6)
plt.title("Precio vs Ancho de pantalla (px)")
plt.show()

# Precio vs Peso
plt.figure(figsize=(6,4))
sns.scatterplot(x='Weight_kg', y='Price_euros', data=df, alpha=0.6)
plt.title("Precio vs Peso (kg)")
plt.show()

# Precio medio por marca
plt.figure(figsize=(10,5))
sns.barplot(x='Company', y='Price_euros', data=df, estimator=np.mean, order=df.groupby('Company')['Price_euros'].mean().sort_values(ascending=False).index)
plt.xticks(rotation=45)
plt.title("Precio medio por marca")
plt.show()
