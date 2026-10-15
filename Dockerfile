FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home retrain \
    && mkdir /state && chown retrain:retrain /state
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=retrain:retrain retrain_app ./retrain_app
USER retrain
ENV STATE_DIR=/state
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "retrain_app.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
