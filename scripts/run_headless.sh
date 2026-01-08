#!/bin/bash
# ============================================================================
# Headless Execution Wrapper for CI/CD Pipeline
# ============================================================================
# This script initializes a virtual display (Xvfb) and executes the digital
# twin simulation or test suite without requiring a physical display.
#
# Usage:
#   ./run_headless.sh pytest core/ci_tests/test_regression.py -v
#   ./run_headless.sh python -m lasercom_digital_twin.runner --fidelity L4
#
# Environment Variables:
#   DISPLAY: X11 display number (default: :99)
#   XVFB_RESOLUTION: Virtual display resolution (default: 1920x1080x24)
#   MUJOCO_GL: MuJoCo rendering backend (default: osmesa)
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Configuration
# ============================================================================
DISPLAY_NUM=${DISPLAY:-:99}
XVFB_RESOLUTION=${XVFB_RESOLUTION:-1920x1080x24}
XVFB_ARGS="-screen 0 ${XVFB_RESOLUTION} -ac +extension GLX +render -noreset"
MUJOCO_GL=${MUJOCO_GL:-osmesa}
PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

# Logging
LOG_PREFIX="[HEADLESS]"

# ============================================================================
# Functions
# ============================================================================
log_info() {
    echo "${LOG_PREFIX} INFO: $1"
}

log_error() {
    echo "${LOG_PREFIX} ERROR: $1" >&2
}

log_success() {
    echo "${LOG_PREFIX} SUCCESS: $1"
}

cleanup() {
    log_info "Cleaning up virtual display..."
    if [ ! -z "${XVFB_PID:-}" ]; then
        kill ${XVFB_PID} 2>/dev/null || true
        log_info "Xvfb process (PID: ${XVFB_PID}) terminated"
    fi
}

# ============================================================================
# Main Execution
# ============================================================================
log_info "========================================="
log_info "MicroPrecisionGimbal Headless Execution"
log_info "========================================="
log_info "Display: ${DISPLAY_NUM}"
log_info "Resolution: ${XVFB_RESOLUTION}"
log_info "MuJoCo GL Backend: ${MUJOCO_GL}"
log_info "PyOpenGL Platform: ${PYOPENGL_PLATFORM}"
log_info "========================================="

# Set environment variables for headless operation
export DISPLAY=${DISPLAY_NUM}
export MUJOCO_GL=${MUJOCO_GL}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}

# Disable interactive matplotlib backend
export MPLBACKEND=Agg

# Register cleanup function
trap cleanup EXIT INT TERM

# ============================================================================
# Start Virtual Display (Xvfb)
# ============================================================================
log_info "Starting Xvfb virtual display..."

# Check if Xvfb is already running
if xdpyinfo -display ${DISPLAY_NUM} >/dev/null 2>&1; then
    log_info "Xvfb already running on ${DISPLAY_NUM}"
else
    # Start Xvfb in background
    Xvfb ${DISPLAY_NUM} ${XVFB_ARGS} &
    XVFB_PID=$!
    
    # Wait for display to be ready (max 10 seconds)
    log_info "Waiting for display to initialize (PID: ${XVFB_PID})..."
    for i in {1..20}; do
        if xdpyinfo -display ${DISPLAY_NUM} >/dev/null 2>&1; then
            log_success "Virtual display ready"
            break
        fi
        if [ $i -eq 20 ]; then
            log_error "Xvfb failed to start after 10 seconds"
            exit 1
        fi
        sleep 0.5
    done
fi

# ============================================================================
# Verify MuJoCo Setup
# ============================================================================
log_info "Verifying MuJoCo installation..."
python3 -c "
import mujoco
import sys
print(f'MuJoCo version: {mujoco.__version__}')
print(f'MuJoCo GL backend: ${MUJOCO_GL}')
sys.exit(0)
" || {
    log_error "MuJoCo verification failed"
    exit 1
}
log_success "MuJoCo verified"

# ============================================================================
# Execute Command
# ============================================================================
if [ $# -eq 0 ]; then
    log_error "No command provided"
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 pytest core/ci_tests/test_regression.py -v"
    exit 1
fi

log_info "Executing command: $@"
log_info "========================================="

# Execute the provided command with all arguments
"$@"
EXIT_CODE=$?

# ============================================================================
# Report Results
# ============================================================================
log_info "========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    log_success "Command completed successfully (exit code: ${EXIT_CODE})"
else
    log_error "Command failed with exit code: ${EXIT_CODE}"
fi
log_info "========================================="

exit ${EXIT_CODE}
