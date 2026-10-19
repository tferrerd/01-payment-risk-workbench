FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ARG REVISION=local
LABEL org.opencontainers.image.title="Payment Risk Workbench" \
      org.opencontainers.image.revision="${REVISION}" \
      io.payment-risk.release="lightgbm-low-margin-dev-v1-explorer-v2" \
      io.payment-risk.model-sha256="05dc10d01d58c57c8b8c6b0ed0d8b042ed143da8e5163ec28914fc8422063b4c"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY requirements-inference.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-inference.txt

COPY payment_risk/ ./payment_risk/
COPY models/candidates/lightgbm-low-margin-dev-v1.joblib models/candidates/lightgbm-low-margin-dev-v1.json ./models/candidates/
COPY deployment/release.json ./deployment/release.json
COPY deployment/explorer-v1.json ./deployment/explorer-v1.json
COPY deployment/explorer-v2.json ./deployment/explorer-v2.json
COPY examples/predict-request.json examples/predict-expected.json ./examples/
COPY examples/explorer-sample.json ./examples/explorer-sample.json
COPY web/ ./web/
COPY scripts/check_release.py ./scripts/check_release.py
COPY scripts/check_operations.py ./scripts/check_operations.py

USER 10001:10001
RUN python scripts/check_release.py --manifest deployment/explorer-v2.json \
    && python -c 'from payment_risk import PaymentRiskPipeline; PaymentRiskPipeline.load()'

ENV APP_REVISION=${REVISION} \
    OMP_THREAD_LIMIT=1 \
    OMP_WAIT_POLICY=PASSIVE \
    GOMP_SPINCOUNT=0

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close()"]
CMD ["python", "-m", "uvicorn", "payment_risk.explorer:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
