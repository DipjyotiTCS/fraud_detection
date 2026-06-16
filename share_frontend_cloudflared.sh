#!/usr/bin/env bash
set -euo pipefail

FRONTEND_PORT=${FRONTEND_PORT:-5173}

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install it first, then rerun this script."
  echo "Linux example:"
  echo "  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared"
  echo "  chmod +x cloudflared"
  echo "  sudo mv cloudflared /usr/local/bin/cloudflared"
  exit 1
fi

echo "Creating public tunnel for frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "Share the generated https://*.trycloudflare.com URL."
cloudflared tunnel --url "http://127.0.0.1:${FRONTEND_PORT}"
