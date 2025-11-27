import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import warnings

# --- 0. CONFIGURACIÓN ---
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# --- 1. CARGAR MODELOS ---
print(f"Directorio actual: {os.getcwd()}")

try:
    # Cargar Scaler
    path_scaler = 'models/scaler_rps.pkl'
    if not os.path.exists(path_scaler): path_scaler = 'scaler_rps.pkl'
    scaler = joblib.load(path_scaler)
    
    # Cargar Modelo (Prioridad al Mejorado)
    if os.path.exists('models/mlp_mejorado.keras'):
        path_keras = 'models/mlp_mejorado.keras'
        print(f"✨ Usando modelo MEJORADO: {path_keras}")
    elif os.path.exists('models/mlp.keras'):
        path_keras = 'models/mlp.keras'
        print(f"⚠️ Usando modelo BASE: {path_keras}")
    else:
        path_keras = 'mlp.keras'

    mlp_model = tf.keras.models.load_model(path_keras)
    
    # Detectar número de clases
    try: num_clases = mlp_model.output_shape[-1]
    except: num_clases = mlp_model.layers[-1].units

    # --- ETIQUETAS CORREGIDAS (Según tu diagnóstico) ---
    if num_clases == 4:
        CLASES = ["Papel", "Piedra", "Tijeras", "Nada"] # Orden corregido
    elif num_clases == 3:
        CLASES = ["Papel", "Piedra", "Tijeras"]
    else:
        CLASES = [f"Gesto {i}" for i in range(num_clases)]
        
    print(f"📋 Etiquetas: {CLASES}")

except Exception as e:
    print(f"❌ Error fatal: {e}")
    exit()

# --- 3. CONFIGURAR MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- 4. LÓGICA DE GANADOR ---
def determinar_ganador(p1, p2):
    p1, p2 = p1.split(" ")[0].lower(), p2.split(" ")[0].lower()
    if p1 == p2: return "EMPATE"
    
    # Reglas (ajustadas a tus nombres)
    wins = [('piedra','tijeras'), ('papel','piedra'), ('tijeras','papel')]
    
    if (p1, p2) in wins: return "GANA JUGADOR 1"
    if (p2, p1) in wins: return "GANA JUGADOR 2"
    return "..."

# --- 5. BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

font = cv2.FONT_HERSHEY_SIMPLEX
color_p1 = (255, 50, 50)   # Azul
color_p2 = (50, 50, 255)   # Rojo
color_neutral = (200, 200, 200)

print("\n🎥 ¡A JUGAR! Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Espejo activado (obligatorio si entrenaste así)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    move_p1 = None
    move_p2 = None
    text_p1 = ""
    text_p2 = ""

    # Interfaz
    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv2.putText(frame, "JUGADOR 1", (50, 50), font, 1, color_p1, 2)
    cv2.putText(frame, "JUGADOR 2", (w-250, 50), font, 1, color_p2, 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- PREPROCESAMIENTO CORRECTO (Normalizado) ---
            wrist = hand_landmarks.landmark[0]
            row = []
            for lm in hand_landmarks.landmark:
                # Restar muñeca (Normalización relativa)
                row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            X = np.array([row])

            try:
                # 1. Escalar
                try: X_scaled = scaler.transform(X)
                except: pass

                # 2. Predecir
                y_proba = mlp_model.predict(X_scaled, verbose=0)[0]
                idx = np.argmax(y_proba)
                conf = y_proba[idx]
                
                label_name = CLASES[idx] if idx < len(CLASES) else "?"
                
                # Filtros (Ignorar "Nada" o confianza baja)
                es_valido = "Nada" not in label_name and "Fondo" not in label_name and conf > 0.5
                
                label_display = f"{label_name} {conf:.0%}"

                # Asignar Jugador (Posición real en pantalla)
                wrist_x = wrist.x
                text_x = int(wrist_x * w) - 50
                text_y = int(wrist.y * h) - 20

                if wrist_x < 0.5: # P1
                    cv2.putText(frame, label_display, (text_x, text_y), font, 0.8, color_p1 if es_valido else color_neutral, 2)
                    if es_valido:
                        move_p1 = label_name
                        text_p1 = label_display
                else: # P2
                    cv2.putText(frame, label_display, (text_x, text_y), font, 0.8, color_p2 if es_valido else color_neutral, 2)
                    if es_valido:
                        move_p2 = label_name
                        text_p2 = label_display

            except Exception: pass

    # Mostrar jugada en marcador superior
    if text_p1: cv2.putText(frame, text_p1, (50, 100), font, 1.2, color_p1, 2)
    if text_p2: cv2.putText(frame, text_p2, (w-250, 100), font, 1.2, color_p2, 2)

    # Resultado central
    if move_p1 and move_p2:
        res = determinar_ganador(move_p1, move_p2)
        cv2.rectangle(frame, (w//2 - 200, h//2 - 40), (w//2 + 200, h//2 + 40), (0,0,0), -1)
        cv2.putText(frame, res, (w//2 - 180, h//2 + 10), font, 1.2, (0, 255, 0), 3)

    cv2.imshow('RPS VS Mode (Final)', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()