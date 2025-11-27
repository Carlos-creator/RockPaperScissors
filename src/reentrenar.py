import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import warnings

# --- CONFIGURACIÓN ---
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

ARCHIVO_NUEVOS = "nuevos_datos.csv"
MODELO_BASE = "models/mlp.keras"          # El original de tu amigo
MODELO_MEJORADO = "models/mlp_mejorado.keras" # Tu versión evolucionada
SCALER_PATH = "models/scaler_rps.pkl"

# 1. CARGAR DATOS NUEVOS
if not os.path.exists(ARCHIVO_NUEVOS):
    print("❌ No hay datos nuevos (nuevos_datos.csv). Ejecuta capturar.py primero.")
    exit()

print("📂 Cargando nuevos datos...")
df = pd.read_csv(ARCHIVO_NUEVOS, header=None)

# Limpieza y validación de datos (Anti-errores)
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
valid_labels = [0, 1, 2, 3] # Aseguramos que solo sean clases válidas
df = df[df.iloc[:, -1].isin(valid_labels)]

print(f"   -> {len(df)} ejemplos nuevos válidos listos para enseñar.")

data = df.values
X_new = data[:, :-1]
y_new = data[:, -1]

# 2. CARGAR SCALER
if not os.path.exists(SCALER_PATH):
    SCALER_PATH = "scaler_rps.pkl" # Fallback raíz

print(f"⚖️  Cargando scaler de: {SCALER_PATH}")
try:
    scaler = joblib.load(SCALER_PATH)
    X_new_scaled = scaler.transform(X_new)
except Exception as e:
    print(f"❌ Error con el scaler: {e}")
    exit()

# 3. SELECCIÓN INTELIGENTE DEL MODELO (Memoria)
if os.path.exists(MODELO_MEJORADO):
    print(f"🚀 ENCONTRADO MODELO PREVIO: {MODELO_MEJORADO}")
    print("   Seguiremos entrenando sobre este para no perder lo aprendido.")
    modelo_a_cargar = MODELO_MEJORADO
else:
    print(f"🔰 PRIMERA VEZ: Cargando modelo base: {MODELO_BASE}")
    modelo_a_cargar = MODELO_BASE

if not os.path.exists(modelo_a_cargar):
    modelo_a_cargar = "mlp.keras" # Fallback raíz

print(f"   Cargando cerebro: {modelo_a_cargar} ...")
model = tf.keras.models.load_model(modelo_a_cargar)

# 4. FINE-TUNING (MODO AGRESIVO / TURBO)
print("\n🧠 RE-ENTRENANDO (Modo Agresivo)...")

# AUMENTAMOS EL LEARNING RATE (0.001 es 10x más fuerte que antes)
# Esto le dice al modelo: "Olvida un poco lo viejo y créeme a mí ahora"
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# AUMENTAMOS LAS ÉPOCAS (50 vueltas)
# Repetimos la lección 50 veces para que se le grabe a fuego.
history = model.fit(X_new_scaled, y_new, epochs=50, batch_size=8, shuffle=True, verbose=0)

# Mostrar resultados
acc_final = history.history['accuracy'][-1]
loss_final = history.history['loss'][-1]

print(f"   ✅ Entrenamiento finalizado.")
print(f"   -> Precisión en tus datos nuevos: {acc_final:.1%}")
print(f"   -> Error final: {loss_final:.4f}")

# 5. GUARDAR
print(f"\n💾 Guardando evolución en: {MODELO_MEJORADO}")
model.save(MODELO_MEJORADO)
print("¡Listo! Tu IA ha aprendido la lección.")