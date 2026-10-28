FROM python:3.12-slim

# DejaVu: fonts for grid.png and captions.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Includes output/: the 18 demo results are committed, not generated here.
COPY . .

ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 OPENCV_LOG_LEVEL=ERROR
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
