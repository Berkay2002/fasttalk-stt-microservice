#!/bin/bash
# Quick start script for STT service on CPU (Linux/Mac)

set -e

echo "===================================================================="
echo "FastTalk STT Service - CPU Mode Quick Start"
echo "===================================================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    # Set CPU as default
    sed -i.bak 's/COMPUTE_DEVICE=cuda/COMPUTE_DEVICE=cpu/' .env
    sed -i.bak 's/WHISPER_MODEL=large-v3/WHISPER_MODEL=base/' .env
    rm .env.bak
    echo "Please edit .env with your configuration"
    echo "Note: COMPUTE_DEVICE is set to 'cpu', WHISPER_MODEL is set to 'base'"
    exit 1
fi

# Load environment
source .env

echo "Configuration:"
echo "  Compute Device: ${COMPUTE_DEVICE:-cpu}"
echo "  Model: ${WHISPER_MODEL:-base}"
echo "  Port: ${STT_PORT:-8001}"
echo "  CPU Threads: ${NUM_THREADS:-12}"
echo "===================================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install CPU version of PyTorch
echo "Installing CPU-optimized dependencies..."
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper==1.1.0
pip install \
    ffmpeg-python \
    soundfile \
    numpy \
    webrtcvad \
    pyaudio \
    flask \
    psutil \
    requests \
    fastapi \
    uvicorn[standard] \
    websockets \
    python-multipart \
    scipy \
    pydub \
    aiofiles \
    python-dotenv

# Set CPU-specific environment variables
export COMPUTE_DEVICE=cpu
export OMP_NUM_THREADS=${NUM_THREADS:-12}
export MKL_NUM_THREADS=${NUM_THREADS:-12}
export OPENBLAS_NUM_THREADS=${NUM_THREADS:-12}

# Start service
echo "Starting STT service in CPU mode..."
python main.py websocket

deactivate
