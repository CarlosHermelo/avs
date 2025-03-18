FROM python:3.9-slim
WORKDIR /app
# Instalar dependencias del sistema necesarias para compilar extensiones nativas
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
# Crear directorios para templates y archivos estáticos
#RUN mkdir -p /app/templates /app/static/css /app/flask_session
# Actualizar pip
RUN pip install --no-cache-dir --upgrade pip
# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flask gunicorn flask-session
# Copiar el código fuente
COPY . .
# Asegurarse de que los directorios existan
#RUN mkdir -p /app/templates /app/static/css /app/flask_session
EXPOSE 5000
# Usar Flask directamente para desarrollo en lugar de gunicorn
CMD ["python", "app.py"]