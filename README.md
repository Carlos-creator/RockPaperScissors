# ✌️ Piedra, Papel o Tijeras - IA con Visión Artificial

  

Este proyecto es una implementación del clásico juego "Piedra, Papel o Tijeras" utilizando **Visión Artificial** y **Deep Learning**.

  

El sistema es capaz de detectar manos en tiempo real mediante una cámara web, extraer sus puntos clave (landmarks) y clasificar el gesto utilizando una Red Neuronal entrenada. Además, incluye un sistema de **Aprendizaje Activo (Active Learning)** que permite re-entrenar y mejorar el modelo capturando nuevos datos personalizados.

  

## 🚀 Características Principales

  

* **Detección de Manos en Tiempo Real:** Utiliza **MediaPipe** para un seguimiento rápido y preciso de la mano.

* **Modo Versus (VS):** Pantalla dividida para jugar contra un amigo o contra la IA (Jugador 1 vs Jugador 2).

* **Deep Learning:** Utiliza un modelo **MLP (Multi-Layer Perceptron)** implementado en **TensorFlow/Keras**.

* **Sistema Robusto:** Normalización de coordenadas relativa a la muñeca para detectar gestos en cualquier posición de la pantalla.

* **Mejora Continua:** Scripts incluidos para capturar errores y re-entrenar el modelo (Fine-Tuning) en segundos.

  

## 📂 Estructura del Proyecto

  

```text

mi-proyecto-rps/

│

├── models/                  # Modelos entrenados y escaladores

│   ├── mlp.keras            # Modelo base de Red Neuronal

│   ├── mlp_mejorado.keras   # Modelo re-entrenado (se genera automáticamente)

│   └── scaler_rps.pkl       # Escalador de datos (StandardScaler)

│

├── src/                     # Código fuente

│   ├── jugar_keras_vs.py    # [PRINCIPAL] Juego en modo VS con modelo Keras

│   ├── capturar.py          # Herramienta para capturar nuevos datos

│   ├── reentrenar.py        # Script para aplicar Fine-Tuning al modelo

│   └── diagnostico.py       # Herramienta visual para depurar la detección

│

├── nuevos_datos.csv         # Base de datos incremental (se genera al capturar)

├── requirements.txt         # Lista de dependencias

└── README.md                # Documentación


```

## ⚙️ Instalación

  

Recomendamos usar un entorno virtual para evitar conflictos de versiones.

  

1.  **Clonar o descargar este repositorio.**

2.  **Crear un entorno virtual:**

    ```bash

    python -m venv .venv

    ```

3.  **Activar el entorno:**

      * Windows (CMD): `.venv\Scripts\activate`

      * Linux/Mac: `source .venv/bin/activate`

4.  **Instalar dependencias:**

    ```bash

    pip install -r requirements.txt

    ```

    *(Si no tienes el archivo, instala manualmente: `pip install opencv-python mediapipe tensorflow scikit-learn pandas numpy joblib`)*

  

## 🎮 Cómo Jugar

  

El script principal utiliza el modelo más avanzado disponible (`mlp_mejorado.keras` si existe, o `mlp.keras` por defecto).

  

Ejecuta el siguiente comando:

  

```bash

python src/jugar_keras_vs.py

```

  

  * **Jugador 1:** Lado izquierdo de la pantalla.

  * **Jugador 2:** Lado derecho de la pantalla.

  * **Salir:** Presiona la tecla `q`.

  

## 🧠 Ciclo de Mejora (Aprendizaje Activo)

  

Si sientes que el modelo falla con ciertos gestos o con tu iluminación, puedes enseñarle para que mejore:

  

1.  **Capturar Datos:**

    Ejecuta `python src/capturar.py`.

  

      * Haz el gesto deseado frente a la cámara.

      * Presiona `0` (Papel), `1` (Piedra) o `2` (Tijeras) repetidamente para guardar ejemplos.

      * *Tip: Mueve ligeramente la mano y varía el ángulo para capturar datos robustos.*

  

2.  **Re-entrenar:**

    Ejecuta `python src/reentrenar.py`.

  

      * El script tomará tus nuevos datos y ajustará los pesos del modelo actual.

      * Generará/Actualizará el archivo `models/mlp_mejorado.keras`.

  

3.  **Jugar:**

    Vuelve a ejecutar el juego. ¡La IA ahora reconocerá mejor tus gestos\!

  

## 🛠️ Herramientas de Diagnóstico

  

Si tienes dudas sobre qué está viendo la IA, usa el modo diagnóstico:

  

```bash

python src/diagnostico.py

```

  

  * Te permite activar/desactivar el escalado (`s`), la normalización de muñeca (`n`) y el modo espejo (`m`) en tiempo real para encontrar la configuración óptima.

  

## 📋 Requisitos Técnicos

  

  * Python 3.8 - 3.12

  * Cámara Web

  

-----

  

*Desarrollado para el curso INF395 - Introducción al Deep Learning.*

  