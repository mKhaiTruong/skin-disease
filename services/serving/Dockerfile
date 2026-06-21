FROM python:3.12-slim

WORKDIR /app

COPY services/serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY packages/vision ./packages/vision
RUN pip install --no-cache-dir ./packages/vision

COPY packages/core ./packages/core
RUN pip install --no-cache-dir ./packages/core

COPY services/serving ./services/serving
COPY config ./config
COPY params.yaml .

EXPOSE 7860

CMD ["uvicorn", "services.serving.main:app", "--host", "0.0.0.0", "--port", "7860"]