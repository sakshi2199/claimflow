# Build context is the repository root (see docker-compose.yml) so that data/ can be copied in.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Layout mirrors the repo (/srv/backend, /srv/data) because config.py locates data/ relative to the code.
WORKDIR /srv/backend

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY data /srv/data

RUN useradd --create-home appuser \
    && mkdir -p /srv/evaluation_results \
    && chown -R appuser /srv/evaluation_results /srv/data
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
