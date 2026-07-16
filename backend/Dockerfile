FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/backend

WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ragpilot ./ragpilot

EXPOSE 8000

CMD ["uvicorn", "ragpilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
