# ---- Stage 1: build the React frontend with Node ----
FROM node:22-slim AS frontend-build
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: the Python runtime image ----
FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# CPU-only PyTorch first, to avoid shipping ~2GB of unused CUDA libraries
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# then the rest of the dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# backend code and the source document
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/

# bring in ONLY the built frontend from stage 1
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]