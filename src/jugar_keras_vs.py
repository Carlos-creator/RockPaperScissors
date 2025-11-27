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
    
    # Cargar Modelo Keras
    path_keras = 'models/mlp_mejorado.keras'
    if not os.path.exists(path_keras): path_keras = 'mlp.keras'
    mlp_model = tf.keras.models.load_model(path_keras)
    
    # Detección de clases
    try:
        num_clases = mlp_model.output_shape[-1]
    except:
        num_clases = mlp_model.layers[-1].units
        
    print(f"✅ Modelos cargados. Detectando {num_clases} clases.")

except Exception as e:
    print(f"❌ Error fatal cargando modelos: {e}")
    exit()

# --- 2. ETIQUETAS ---
if num_clases == 4:
    CLASES = ["Nada", "Papel", "Piedra", "Tijeras"]
elif num_clases == 3:
    CLASES = ["Papel", "Piedra", "Tijeras"]
else:
    CLASES = [f"Gesto {i}" for i in range(num_clases)]

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

print("\n🎥 Cámara iniciada. Presiona 'q' en la ventana para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    move_p1 = None
    move_p2 = None
    text_p1 = ""
    text_p2 = ""

    # Interfaz base
    cv2.line(frame, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv2.putText(frame, "JUGADOR 1", (50, 50), font, 1, color_p1, 2)
    cv2.putText(frame, "JUGADOR 2", (w-250, 50), font, 1, color_p2, 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # === AQUÍ ESTÁ LA MAGIA DE LA NORMALIZACIÓN ===
            landmarks = hand_landmarks.landmark
            wrist = landmarks[0] # La muñeca es el origen (0,0,0) relativo
            
            row = []
            for lm in landmarks:
                # Restamos la posición de la muñeca a cada punto
                # Así el gesto es igual sin importar dónde esté en la pantalla
                row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            # Convertimos a DataFrame con índices numéricos para el scaler
            X = pd.DataFrame([row], columns=range(len(row)))

            try:
                # Predicción
                X_scaled = scaler.transform(X)
                y_proba = mlp_model.predict(X_scaled, verbose=0)[0]
                idx = np.argmax(y_proba)
                conf = y_proba[idx]
                
                label_name = CLASES[idx] if idx < len(CLASES) else "?"
                
                # Filtros de visualización
                es_valido = "Nada" not in label_name and "Fondo" not in label_name and conf > 0.4
                
                label_display = f"{label_name} {conf:.0%}"

                # Asignar Jugador (Usamos la posición real en pantalla, no la normalizada)
                wrist_screen_x = wrist.x 
                text_x = int(wrist_screen_x * w) - 50
                text_y = int(wrist.y * h) - 20

                if wrist_screen_x < 0.5: # Jugador 1
                    cv2.putText(frame, label_display, (text_x, text_y), font, 0.8, color_p1 if es_valido else color_neutral, 2)
                    if es_valido:
                        move_p1 = label_name
                        text_p1 = label_display
                else: # Jugador 2
                    cv2.putText(frame, label_display, (text_x, text_y), font, 0.8, color_p2 if es_valido else color_neutral, 2)
                    if es_valido:
                        move_p2 = label_name
                        text_p2 = label_display

            except Exception as e:
                pass # Ignoramos errores puntuales para no congelar el video

    # Marcador superior
    if text_p1: cv2.putText(frame, text_p1, (50, 100), font, 1.2, color_p1, 2)
    if text_p2: cv2.putText(frame, text_p2, (w-250, 100), font, 1.2, color_p2, 2)

    # Resultado central
    if move_p1 and move_p2:
        res = determinar_ganador(move_p1, move_p2)
        cv2.rectangle(frame, (w//2 - 200, h//2 - 40), (w//2 + 200, h//2 + 40), (0,0,0), -1)
        cv2.putText(frame, res, (w//2 - 180, h//2 + 10), font, 1.2, (0, 255, 0), 3)

    cv2.imshow('RPS Keras Normalizado', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()