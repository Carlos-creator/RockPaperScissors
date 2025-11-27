import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# --- CONFIGURACIÓN ---
ARCHIVO_DATOS = "nuevos_datos.csv"
# ¡OJO! Asegúrate que este orden coincida con tu CLASES en jugar_keras.py
MAPA_TECLAS = {
    ord('0'): 0, # Tecla 0 -> Clase 0 (Ej: Papel)
    ord('1'): 1, # Tecla 1 -> Clase 1 (Ej: Piedra)
    ord('2'): 2, # Tecla 2 -> Clase 2 (Ej: Tijeras)
}
NOMBRES = {0: "Papel", 1: "Piedra", 2: "Tijeras"}

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print(f"\n=== MODO CAPTURA DE DATOS ===")
print(f"Pon tu mano y presiona:")
print(f" '0' para guardar como {NOMBRES[0]}")
print(f" '1' para guardar como {NOMBRES[1]}")
print(f" '2' para guardar como {NOMBRES[2]}")
print(f" 'q' para Salir\n")

datos_guardados = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    status_text = "Listo para capturar..."
    color_status = (255, 255, 255)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- INTERACCIÓN ---
            key = cv2.waitKey(1) & 0xFF
            
            if key in MAPA_TECLAS:
                clase_id = MAPA_TECLAS[key]
                
                # 1. Extraer y Normalizar (IGUAL QUE EN EL JUEGO)
                landmarks = hand_landmarks.landmark
                wrist = landmarks[0]
                row = []
                for lm in landmarks:
                    row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                # 2. Agregar la etiqueta al final
                row.append(clase_id)
                
                # 3. Guardar en memoria
                datos_guardados.append(row)
                
                print(f"📸 Capturado: {NOMBRES[clase_id]} ({len(datos_guardados)} total)")
                status_text = f"GUARDADO: {NOMBRES[clase_id]}"
                color_status = (0, 255, 0)

    # Interfaz
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
    cv2.imshow('Capturador de Datos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- GUARDAR EN CSV AL SALIR ---
if datos_guardados:
    df = pd.DataFrame(datos_guardados)
    # Si el archivo no existe, lo crea con header. Si existe, agrega sin header.
    header = not os.path.exists(ARCHIVO_DATOS)
    df.to_csv(ARCHIVO_DATOS, mode='a', header=header, index=False)
    print(f"\n✅ Se guardaron {len(datos_guardados)} nuevos ejemplos en {ARCHIVO_DATOS}")
else:
    print("\n⚠️ No se guardó nada.")

cap.release()
cv2.destroyAllWindows()