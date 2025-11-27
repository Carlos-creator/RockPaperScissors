import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import tensorflow as tf
import warnings

# --- 0. CONFIGURACIÓN Y LIMPIEZA ---
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# --- 1. CARGAR MODELOS ---
print("Cargando modelos...")
try:
    # Buscar Scaler
    path_scaler = 'models/scaler_rps.pkl'
    if not os.path.exists(path_scaler): path_scaler = 'scaler_rps.pkl'
    scaler = joblib.load(path_scaler)
    
    # Buscar Modelo
    path_keras = 'models/mlp.keras'
    if not os.path.exists(path_keras): path_keras = 'mlp.keras'
    mlp_model = tf.keras.models.load_model(path_keras)
    
    # Detectar clases
    try:
        num_clases = mlp_model.output_shape[-1]
    except:
        num_clases = mlp_model.layers[-1].units

    if num_clases == 4: CLASES = ["Nada", "Papel", "Piedra", "Tijeras"]
    elif num_clases == 3: CLASES = ["Papel", "Piedra", "Tijeras"]
    else: CLASES = [f"C{i}" for i in range(num_clases)]
    
    print("✅ Modelos cargados.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Verifica que 'scaler_rps.pkl' y 'mlp.keras' estén en la carpeta 'models/'")
    exit()

# --- 2. MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- 3. VARIABLES DE CONTROL ---
usar_scaler = True      # Tecla 's'
usar_norm_muñeca = True # Tecla 'n' (Resta la posición de la muñeca)
usar_espejo = True      # Tecla 'm'

# --- 4. BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

font = cv2.FONT_HERSHEY_SIMPLEX

print("\n=== CONTROLES ===")
print(" [s] : Activar/Desactivar SCALER")
print(" [n] : Activar/Desactivar RESTA DE MUÑECA (Normalización)")
print(" [m] : Activar/Desactivar MODO ESPEJO")
print(" [q] : Salir")
print("=================\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Aplicar Espejo
    if usar_espejo:
        frame = cv2.flip(frame, 1)
    
    # --- CORRECCIÓN: Calcular dimensiones SIEMPRE aquí ---
    h, w, _ = frame.shape
    # -----------------------------------------------------

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Estado para mostrar en pantalla
    estado_str = f"Scaler: {'ON' if usar_scaler else 'OFF'} | " \
                 f"Munyeca: {'ON' if usar_norm_muñeca else 'OFF'} | " \
                 f"Espejo: {'ON' if usar_espejo else 'OFF'}"
    
    pred_text = "..."
    conf_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 2. Extraer Datos
            wrist = hand_landmarks.landmark[0]
            row = []
            for lm in hand_landmarks.landmark:
                if usar_norm_muñeca:
                    # Normalización relativa (Resta de muñeca)
                    row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                else:
                    # Coordenadas absolutas
                    row.extend([lm.x, lm.y, lm.z])
            
            X = np.array([row])

            # 3. Aplicar Scaler (Opcional)
            if usar_scaler:
                try:
                    X = scaler.transform(X)
                except:
                    pass 

            # 4. Predecir
            try:
                y_proba = mlp_model.predict(X, verbose=0)[0]
                idx = np.argmax(y_proba)
                conf = y_proba[idx]
                
                label = CLASES[idx] if idx < len(CLASES) else "?"
                
                # Color según confianza
                if conf > 0.5: color = (0, 255, 0) 
                else: color = (0, 165, 255)

                pred_text = label.upper()
                conf_text = f"{conf:.0%}"
                
                # --- VISUALIZACIÓN DE BARRAS ---
                y_offset = 150
                for i, prob in enumerate(y_proba):
                    nom = CLASES[i] if i < len(CLASES) else str(i)
                    barra = int(prob * 100)
                    # Fondo gris
                    cv2.rectangle(frame, (100, y_offset-15), (200, y_offset), (50,50,50), -1)
                    # Barra verde
                    cv2.rectangle(frame, (100, y_offset-15), (100+barra, y_offset), (0,255,0), -1)
                    # Texto
                    cv2.putText(frame, f"{nom}: {barra}%", (10, y_offset), font, 0.5, (200,200,200), 1)
                    y_offset += 25
                    
            except Exception as e:
                pred_text = "Error"
                # print(e) # Descomentar si quieres ver el error en consola

            # Texto sobre la mano
            tx, ty = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)
            cv2.putText(frame, f"{pred_text} {conf_text}", (tx, ty-20), font, 1, color, 2)

    # Interfaz inferior
    cv2.rectangle(frame, (0,0), (w, 80), (0,0,0), -1)
    cv2.putText(frame, "MODO DIAGNOSTICO", (10, 30), font, 0.8, (0,255,255), 2)
    cv2.putText(frame, estado_str, (10, 60), font, 0.6, (255,255,255), 1)
    
    cv2.imshow('Diagnostico RPS', frame)

    # Control de Teclas
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): usar_scaler = not usar_scaler
    elif key == ord('n'): usar_norm_muñeca = not usar_norm_muñeca
    elif key == ord('m'): usar_espejo = not usar_espejo

cap.release()
cv2.destroyAllWindows()