FROM python:3.11-slim

# CPU build of PyTorch first, so pip does not pull the multi-GB CUDA wheel for Linux
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-app.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-app.txt

RUN useradd --create-home triage
# Owned by triage before the volume mounts here, so a new volume inherits that owner
RUN mkdir -p /home/triage/.cache/huggingface && chown -R triage:triage /home/triage/.cache
WORKDIR /app
COPY src /app/src
COPY static /app/static
USER triage

EXPOSE 8010
CMD ["uvicorn", "app.main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8010"]
