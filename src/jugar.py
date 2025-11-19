import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd

# 1. CARGAR LOS MODELOS ENTRENADOS
# Es crucial que estos archivos estén en la misma carpeta

# AHORA PON ESTO (busca dentro de la carpeta models):
import os
print("Directorio actual:", os.getcwd()) # Esto te imprimirá dónde está parado Python

# Intentamos cargar desde la carpeta 'models'
try:
    svm_model = joblib.load('models/svm_rps_model.pkl') 
    scaler = joblib.load('models/scaler_rps.pkl')
except FileNotFoundError:
    # Si falla, intentamos buscar en la raíz 
    svm_model = joblib.load('svm_rps_model.pkl')
    scaler = joblib.load('scaler_rps.pkl')


# 2. CONFIGURAR MEDIAPIPE (Para detectar las manos)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Configuración similar a la usada en el notebook de entrenamiento
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# 3. INICIAR LA CÁMARA
cap = cv2.VideoCapture(0) # 0 suele ser la webcam por defecto en Windows

print("Iniciando cámara... Presiona 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer la cámara.")
        break

    # MediaPipe trabaja con RGB, OpenCV con BGR
    # Volteamos la imagen horizontalmente para efecto espejo
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # Detección de manos
    results = hands.process(image)

    # Volver a BGR para mostrar en pantalla con OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    prediction_label = "Esperando mano..."
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los puntos de la mano en pantalla
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 4. PREPROCESAMIENTO (CRUCIAL: Debe ser idéntico al entrenamiento)
            # En tu notebook extrajeron x, y, z de los 21 puntos 
            row = []
            for landmark in hand_landmarks.landmark:
                row.append(landmark.x)
                row.append(landmark.y)
                row.append(landmark.z)

            # Convertir a formato compatible con el modelo (DataFrame de 1 fila)
            # El modelo espera 63 características (21 puntos * 3 coord)
            X = np.array([row])
            
            # 5. ESCALADO Y PREDICCIÓN
            # Usamos el scaler cargado para transformar los datos igual que en el entrenamiento [cite: 192]
            X_scaled = scaler.transform(X)
            
            # Predecir con el modelo SVM [cite: 205]
            prediction_class = svm_model.predict(X_scaled)[0]
            
            # Mostrar probabilidad (opcional)
            proba = svm_model.predict_proba(X_scaled).max()
            
            prediction_label = f"{prediction_class} ({proba:.2f})"

    # 6. MOSTRAR RESULTADO EN PANTALLA
    # Dibujar un rectángulo y el texto
    cv2.rectangle(image, (10, 10), (400, 70), (255, 255, 255), -1)
    cv2.putText(image, f"Jugada: {prediction_label}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Piedra, Papel o Tijeras - Inferencia', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()