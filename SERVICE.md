# Memory Viewer Service

Run the Memory Viewer as a persistent background service that survives terminal closures and starts automatically.

## Quick Start

```bash
# Install and start the service
./install-service.sh

# Enable auto-start on boot (optional)
loginctl enable-linger $USER
```

Access the viewer at **http://localhost:5000** anytime!

## Service Management

```bash
# Check status
systemctl --user status memory-viewer

# Stop service
systemctl --user stop memory-viewer

# Start service
systemctl --user start memory-viewer

# Restart service
systemctl --user restart memory-viewer

# View logs (follow mode)
journalctl --user -u memory-viewer -f

# View last 50 log lines
journalctl --user -u memory-viewer -n 50
```

## Uninstall

```bash
./uninstall-service.sh
```

## How It Works

- **systemd user service**: Runs under your user account (no root needed)
- **Auto-restart**: If the Flask app crashes, systemd restarts it after 10 seconds
- **Persistent**: Survives terminal closures, tmux detaches, SSH disconnections
- **Logging**: All output goes to systemd journal (`journalctl`)

## Auto-Start on Boot

By default, user services stop when you log out. To keep the service running even after logout:

```bash
loginctl enable-linger $USER
```

This enables "linger" mode, which keeps your user services running 24/7.

## Manual Running (Alternative)

If you don't want a service, you can run manually in the background:

```bash
# Run in background with nohup
nohup uv run memory_viewer.py > /tmp/memory-viewer.log 2>&1 &

# Or use tmux/screen
tmux new -d -s memory-viewer 'uv run memory_viewer.py'
```

## Troubleshooting

**Service won't start?**
```bash
# Check logs for errors
journalctl --user -u memory-viewer -n 50

# Verify service file
systemctl --user cat memory-viewer

# Reload systemd if you edited the service file
systemctl --user daemon-reload
systemctl --user restart memory-viewer
```

**Port 5000 already in use?**

Edit `memory_viewer.py` and change the port:
```python
app.run(host="127.0.0.1", port=5001)  # Use 5001 instead
```

Then restart the service:
```bash
systemctl --user restart memory-viewer
```

**Service doesn't auto-start on boot?**

Make sure linger is enabled:
```bash
loginctl show-user $USER | grep Linger
# Should show: Linger=yes

# If not:
loginctl enable-linger $USER
```
