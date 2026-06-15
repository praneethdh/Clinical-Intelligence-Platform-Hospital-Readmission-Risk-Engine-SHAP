# ----------------------------------
# Stage 1: Build the React Frontend
# ----------------------------------
FROM node:18 AS frontend-builder

WORKDIR /app/frontend
# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy the rest of the frontend code and build it
COPY frontend/ ./
RUN npm run build

# ----------------------------------
# Stage 2: Build the FastAPI Backend
# ----------------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy Python requirements and install
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files (models, data, scripts)
COPY backend/ .

# Copy the compiled React build from Stage 1 into the backend folder
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Hugging Face Spaces explicitly require apps to run on port 7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
