# Análisis de Rugosidad Musical en Acordes

Este proyecto contiene una serie de herramientas y notebooks para el análisis de la rugosidad en acordes musicales, utilizando diversos modelos teóricos y técnicas de reducción de dimensionalidad.

---

##  Prerequisites (Prerrequisitos)

Antes de empezar, asegúrate de tener instalado el siguiente software en tu sistema. Estas herramientas son necesarias para preparar el entorno y compilar algunas de las librerías.

1.  **Python (versión 3.9+):** El lenguaje de programación principal.

    * [Descargar Python](https://www.python.org/downloads/)

    * **Nota para Windows:** Durante la instalación, es crucial marcar la casilla **"Add Python to PATH"**.

2.  **Git:** El sistema de control de versiones, necesario para instalar `chordcodex` directamente desde GitHub.

    * [Descargar Git](https://git-scm.com/downloads/)

---

## ⚙️ Installation (Instalación)

Sigue estos pasos para configurar el entorno de desarrollo localmente.

1.  **Clona el repositorio:**

    ```bash

    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)

    ```

2.  **Navega a la carpeta del proyecto:**

    ```bash

    cd tu-repositorio

    ```

3.  **Crea un entorno virtual:**

    *Esto crea una "caja de herramientas" aislada para el proyecto.*

    ```bash

    # En Windows

    py -m venv venv

    # En Mac/Linux

    python3 -m venv venv

    ```

4.  **Activa el entorno virtual:**

    ```bash

    # En Windows

    .\venv\Scripts\activate

    # En Mac/Linux

    source venv/bin/activate

    ```

    *Verás `(venv)` al principio de la línea de tu terminal si se activó correctamente.*

5.  **Instala todas las librerias necesarias:**

    *Este comando lee el archivo `requirements.txt` e instala todas las dependencias con las versiones exactas para garantizar la compatibilidad.*

    ```bash

    pip install -r requirements.txt

    ```

    *Nota:* La reproduccion de audio en los notebooks funciona con `IPython.display.Audio`, por lo que no requiere librerias nativas adicionales.

6.  **Configura la conexion a la base de datos:**
    *Actualiza el archivo `.env` con los valores de tu servidor PostgreSQL local (host, puerto, usuario, contrasena y nombre de la base).*
    ```bash
    DB_HOST=localhost
    DB_PORT=5432
    DB_USER=tu_usuario
    DB_PASSWORD=tu_password
    DB_NAME=ChordCodex
    ```


---

## 🚀 Running the Project (Ejecutar el Proyecto)

La forma más fiable de ejecutar los notebooks de Jupyter para este proyecto es lanzándolos directamente desde el terminal.

1.  Asegúrate de que tu entorno virtual `(venv)` esté activado.

2.  Ejecuta el siguiente comando en el terminal:

    ```bash

    python -m notebook

    ```

3.  Esto abrirá automáticamente una pestaña en tu navegador web. Desde allí, navega y abre el archivo `.ipynb` que desees ejecutar.

¡Y eso es todo! El entorno está listo para explorar los análisis.

