FROM python:3.11-slim

ARG INCLUDE_LOCAL_MODELS=false

COPY requirements-hosted.txt requirements-app.txt /app/
# Install CPU PyTorch and local Hugging Face packages only when INCLUDE_LOCAL_MODELS=true
RUN if [ "$INCLUDE_LOCAL_MODELS" = "true" ]; then \
      pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r /app/requirements-app.txt; \
    else \
      pip install --no-cache-dir -r /app/requirements-hosted.txt; \
    fi

RUN useradd --create-home triage
# Owned by triage before the volume mounts here, so a new volume inherits that owner
RUN mkdir -p /home/triage/.cache/huggingface && chown -R triage:triage /home/triage/.cache
WORKDIR /app
COPY main.py /app/
COPY app /app/app
COPY config /app/config
COPY static /app/static
USER triage

EXPOSE 8010
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8010 --proxy-headers --no-access-log"]
