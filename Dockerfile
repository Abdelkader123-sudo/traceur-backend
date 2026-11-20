FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py auth.py messagerie.py temps_travail.py colliers.py fcm_notifications.py DBSCAN.py geofence.py params.py emqxsl-ca.crt trajectoire_service.py trajectoire.onnx trajectoire_meta.json ./
COPY IA/ ./IA/

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]