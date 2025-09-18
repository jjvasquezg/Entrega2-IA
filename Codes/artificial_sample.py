# --- Seleccionar el modelo final (Random Forest entrenado previamente) ---
modelo_final = rf  # asumimos que Random Forest fue el elegido

# --- Crear muestra artificial ---
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

# --- Predecir el precio ---
prediccion = modelo_final.predict(nueva_muestra)
print("Predicción de precio en euros:", prediccion[0])
