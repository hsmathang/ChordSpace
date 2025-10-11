# Base ligera con Python 3.11
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias de sistema mínimas para compilar/instalar paquetes y clonar repos (chordcodex vía Git)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential \
  && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero para aprovechar la cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Crear carpeta de salida por defecto
RUN mkdir -p /app/outputs/demo

# Comando por defecto: ejecutar un experimento mínimo reproducible sin DB
CMD ["python", "-m", "tools.run_lab", "--model", "Sethares", "--metric", "cosine", "--reduction", "MDS", "--out", "outputs/demo"]
