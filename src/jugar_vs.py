import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

# --- 1. CARGAR MODELOS ---
print(f"Directorio actual: {os.getcwd()}")
try:
    # Intentamos rutas comunes para evitar errores
    if os.path.exists('models/svm_rps_model.pkl'):
        model_path = 'models/'
    elif os.path.exists('models/svm_rps_model.pkl'):
        model_path = ''
    else:
        raise FileNotFoundError("No se encuentran los archivos .pkl")

    svm_model = joblib.load(os.path.join(model_path, 'svm_rps_model.pkl'))
    scaler = joblib.load(os.path.join(model_path, 'scaler_rps.pkl'))
    print("¡Modelos cargados para VS Mode!")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    exit()

# --- 2. CONFIGURAR MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, # Importante: Permitir 2 manos
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

# --- 3. LÓGICA DEL JUEGO ---
def determinar_ganador(jugada_p1, jugada_p2):
    # Normalizamos a minúsculas para comparar por si acaso
    p1 = jugada_p1.lower()
    p2 = jugada_p2.lower()
    
    if p1 == p2:
        return "EMPATE"
    
    # Reglas clásicas
    # Asumimos que el modelo devuelve "Piedra", "Papel", "Tijeras" 
    # (o en inglés "Rock", "Paper", "Scissors" según cómo lo entrenaron)
    # Ajusta los strings abajo si tu modelo usa otros nombres.
    
    ganador_p1 = (
        (p1 == "piedra" and p2 == "tijeras") or
        (p1 == "papel" and p2 == "piedra") or
        (p1 == "tijeras" and p2 == "papel") or
        # Soporte para inglés por si acaso
        (p1 == "rock" and p2 == "scissors") or
        (p1 == "paper" and p2 == "rock") or
        (p1 == "scissors" and p2 == "paper")
    )
    
    if ganador_p1:
        return "GANA JUGADOR 1"
    else:
        return "GANA JUGADOR 2"

# --- 4. BUCLE PRINCIPAL ---
# Intenta índice 0 o 1 según tu cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 

# Configuración de fuentes y colores
font = cv2.FONT_HERSHEY_SIMPLEX
color_p1 = (255, 50, 50)   # Azulado
color_p2 = (50, 50, 255)   # Rojizo
color_win = (0, 255, 0)    # Verde

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Efecto espejo y conversión de color
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    
    # Convertir a RGB para MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Variables para guardar la jugada de este cuadro
    move_p1 = None
    move_p2 = None

    # --- DIBUJAR LA INTERFAZ ---
    # Línea divisoria central
    cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
    # Etiquetas de Jugadores
    cv2.putText(frame, "JUGADOR 1", (50, 50), font, 1, color_p1, 2, cv2.LINE_AA)
    cv2.putText(frame, "JUGADOR 2", (width - 250, 50), font, 1, color_p2, 2, cv2.LINE_AA)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar esqueleto
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- PREPROCESAMIENTO ---
            row = []
            for landmark in hand_landmarks.landmark:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)
            
            X = np.array([row])
            X_scaled = scaler.transform(X)
            prediction = svm_model.predict(X_scaled)[0]

            # --- DETERMINAR DE QUIÉN ES LA MANO ---
            # Usamos la coordenada X de la muñeca (landmark 0)
            wrist_x = hand_landmarks.landmark[0].x
            
            # Coordenadas para poner el texto en la mano
            text_x = int(hand_landmarks.landmark[0].x * width)
            text_y = int(hand_landmarks.landmark[0].y * height)

            if wrist_x < 0.5:
                # Lado Izquierdo (Jugador 1)
                move_p1 = prediction
                cv2.putText(frame, move_p1, (text_x, text_y), font, 0.8, color_p1, 2)
            else:
                # Lado Derecho (Jugador 2)
                move_p2 = prediction
                cv2.putText(frame, move_p2, (text_x, text_y), font, 0.8, color_p2, 2)

    # --- RESULTADO FINAL ---
    if move_p1 and move_p2:
        resultado = determinar_ganador(move_p1, move_p2)
        
        # Fondo negro para el texto del resultado
        cv2.rectangle(frame, (width//2 - 200, height//2 - 60), 
                             (width//2 + 200, height//2 + 60), (0,0,0), -1)
        
        # Texto resultado
        text_size = cv2.getTextSize(resultado, font, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, resultado, (text_x, height//2 + 10), font, 1.5, color_win, 3)
    
    elif not move_p1 and not move_p2:
        cv2.putText(frame, "Esperando manos...", (width//2 - 100, height - 50), font, 0.7, (200,200,200), 1)

    # Mostrar
    cv2.imshow('Piedra, Papel o Tijeras - VS MODE', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()