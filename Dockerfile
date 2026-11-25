# Hugging Face Spaces Dockerfile for ALOHA Bimanual Toss & Catch
FROM python:3.10-slim

# System packages for headless MuJoCo and off-screen rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgl1-mesa-dri \
    libosmesa6 \
    libosmesa6-dev \
    libglib2.0-0 \
    libglfw3 \
    libglew-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user setup required by Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa

WORKDIR /app

# Install python dependencies
COPY --chown=user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir opencv-python-headless && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY --chown=user . /app

EXPOSE 7860

CMD ["python", "run_live_cockpit.py", "--host", "0.0.0.0", "--port", "7860", "--no-browser"]
