FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY dev-requirements.txt /app/dev-requirements.txt
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/dev-requirements.txt

# Copy project
COPY . /app

# Create model folder - user should mount or copy model to /app/model
RUN mkdir -p /app/model

EXPOSE 5003

# Default runtime environment variables
ENV MODEL_PATH=/app/model/model_final.pth
ENV MODEL_TYPE=pytorch
ENV CLASS_NAMES="contract form invoice letter receipt"
ENV IMG_SIZE=256

CMD ["sh", "-c", "python app.py --model-path ${MODEL_PATH} --model-type ${MODEL_TYPE} --class-names ${CLASS_NAMES} --img-size ${IMG_SIZE} --host 0.0.0.0 --port 5003"]
