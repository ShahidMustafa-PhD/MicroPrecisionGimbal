# ============================================================================
# MicroPrecisionGimbal Digital Twin - Production Docker Image
# ============================================================================
# Multi-stage build optimized for CI/CD pipeline execution
# Supports headless simulation with hardware-accelerated rendering disabled
#
# Usage:
#   docker build -t lasercom-digital-twin:latest .
#   docker run --rm lasercom-digital-twin:latest pytest core/ci_tests/
#
# Build Arguments:
#   PYTHON_VERSION: Python version (default: 3.11)
#   MUJOCO_VERSION: MuJoCo version (default: 3.2.5)
# ============================================================================

# ============================================================================
# Stage 1: Base Environment with System Dependencies
# ============================================================================
FROM python:3.11-slim-bookworm AS base

# Prevent Python from buffering stdout/stderr (critical for CI logs)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for MuJoCo and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    # MuJoCo dependencies
    libgl1-mesa-glx \
    libglew2.2 \
    libosmesa6 \
    libglfw3 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    # Virtual display for headless rendering
    xvfb \
    x11-utils \
    # Build tools (needed for some Python packages)
    gcc \
    g++ \
    make \
    # Utilities
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Configure headless display (Xvfb - X Virtual Frame Buffer)
# This allows MuJoCo to initialize without a physical display
ENV DISPLAY=:99 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

# ============================================================================
# Stage 2: Python Dependencies
# ============================================================================
FROM base AS dependencies

# Set working directory
WORKDIR /app

# Copy requirements file first (layer caching optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional CI/CD specific packages
RUN pip install --no-cache-dir \
    pytest>=7.4.0 \
    pytest-cov>=4.1.0 \
    pytest-xdist>=3.3.0 \
    pytest-timeout>=2.1.0 \
    coverage>=7.3.0

# ============================================================================
# Stage 3: Application Layer
# ============================================================================
FROM dependencies AS application

# Copy entire application codebase
COPY . .

# Create necessary directories
RUN mkdir -p \
    logs \
    results \
    output \
    data/telemetry \
    config

# Set permissions (allow non-root execution)
RUN chmod -R 755 /app && \
    chmod +x scripts/run_headless.sh

# ============================================================================
# Stage 4: Runtime Configuration
# ============================================================================
FROM application AS runtime

# Set working directory
WORKDIR /app

# Expose Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Health check: Verify Python and key dependencies
RUN python -c "import numpy; import scipy; import matplotlib; import mujoco; print('✓ All dependencies loaded successfully')"

# Default entrypoint: Start Xvfb and run tests
# Override with custom commands as needed
ENTRYPOINT ["/app/scripts/run_headless.sh"]

# Default command: Run regression tests
CMD ["pytest", "core/ci_tests/test_regression.py", "-v", "--tb=short", "--strict-markers"]

# ============================================================================
# Build Metadata
# ============================================================================
LABEL maintainer="MicroPrecisionGimbal Team" \
      description="Lasercom Digital Twin - CI/CD Execution Environment" \
      version="1.0.0" \
      aerospace.standard="DO-178C Level B" \
      headless.support="true" \
      mujoco.version="3.2.5"

# ============================================================================
# Usage Examples:
# ============================================================================
# Build:
#   docker build -t lasercom-digital-twin:latest .
#
# Run regression tests:
#   docker run --rm lasercom-digital-twin:latest
#
# Run specific test file:
#   docker run --rm lasercom-digital-twin:latest pytest core/ci_tests/test_regression.py -v
#
# Run all tests:
#   docker run --rm lasercom-digital-twin:latest pytest lasercom_digital_twin/tests/ -v
#
# Interactive shell:
#   docker run --rm -it lasercom-digital-twin:latest bash
#
# Run simulation with specific fidelity:
#   docker run --rm lasercom-digital-twin:latest python -m lasercom_digital_twin.runner --fidelity L4
#
# Mount local results directory:
#   docker run --rm -v $(pwd)/results:/app/results lasercom-digital-twin:latest
# ============================================================================
