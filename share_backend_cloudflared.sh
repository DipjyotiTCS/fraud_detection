#!/usr/bin/env bash
set -euo pipefail

BACKEND_PORT=${BACKEND_PORT:-8000}

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is not installed. Install it first, then rerun this script."
  echo "Linux example:"
  echo "  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared"
  echo "  chmod +x cloudflared"
  echo "  sudo mv cloudflared /usr/local/bin/cloudflared"
  exit 1
fi

echo "Creating public tunnel for backend: http://127.0.0.1:${BACKEND_PORT}"
echo "Copy the generated https://*.trycloudflare.com URL into the React UI Backend URL field."
cloudflared tunnel --url "http://127.0.0.1:${BACKEND_PORT}"
