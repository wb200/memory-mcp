#!/bin/bash
# Uninstall memory-viewer systemd user service

set -e

SERVICE_FILE="memory-viewer.service"
SYSTEMD_DIR="$HOME/.config/systemd/user"

echo "Uninstalling memory-viewer systemd service..."

# Stop and disable service
systemctl --user stop memory-viewer.service 2>/dev/null || true
systemctl --user disable memory-viewer.service 2>/dev/null || true

# Remove service file
rm -f "$SYSTEMD_DIR/$SERVICE_FILE"

# Reload systemd daemon
systemctl --user daemon-reload

echo "✓ Service uninstalled!"
