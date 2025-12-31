#!/bin/bash
# Install memory-viewer as a systemd user service

set -e

SERVICE_FILE="memory-viewer.service"
SYSTEMD_DIR="$HOME/.config/systemd/user"

echo "Installing memory-viewer systemd service..."

# Create systemd user directory if it doesn't exist
mkdir -p "$SYSTEMD_DIR"

# Copy service file
cp "$SERVICE_FILE" "$SYSTEMD_DIR/"

# Reload systemd daemon
systemctl --user daemon-reload

# Enable and start service
systemctl --user enable memory-viewer.service
systemctl --user start memory-viewer.service

echo ""
echo "✓ Service installed and started!"
echo ""
echo "Useful commands:"
echo "  systemctl --user status memory-viewer    # Check status"
echo "  systemctl --user stop memory-viewer      # Stop service"
echo "  systemctl --user start memory-viewer     # Start service"
echo "  systemctl --user restart memory-viewer   # Restart service"
echo "  journalctl --user -u memory-viewer -f    # View logs"
echo ""
echo "To enable auto-start on boot:"
echo "  loginctl enable-linger $USER"
echo ""
echo "Access viewer at: http://localhost:5000"
