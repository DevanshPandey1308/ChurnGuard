FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHURNGUARD_MODEL_DIR=/app/models

WORKDIR /app

# LightGBM's Linux wheel uses the GNU OpenMP runtime.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 churnguard \
    && useradd --system --uid 10001 --gid churnguard --home-dir /nonexistent --shell /usr/sbin/nologin churnguard

COPY pyproject.toml ./
COPY src/churnguard ./src/churnguard
RUN python -m pip install --no-cache-dir .

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import json, urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4); d=json.load(r); assert r.status == 200 and d.get('status') == 'ok' and all(d.get(k) for k in ('churn_model_loaded','calibrator_loaded','future_value_model_loaded'))"

CMD ["uvicorn", "churnguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
