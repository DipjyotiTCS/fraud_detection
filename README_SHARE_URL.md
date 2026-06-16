# Temporary Shareable URL for the React Frontend

The React app performs orchestration in the browser, so a public frontend URL alone is not enough. The browser also needs to reach the FastAPI backend.

Use two tunnels:

1. One tunnel for FastAPI backend: `http://127.0.0.1:8000`
2. One tunnel for React frontend: `http://127.0.0.1:5173`

## Recommended option: Cloudflare Tunnel

Cloudflare Tunnel works well for temporary sharing without opening inbound firewall ports.

### 1. Install cloudflared

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/cloudflared
cloudflared --version
```

### 2. Start backend

From the project root:

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open.

### 3. Start frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev:public
```

Keep this terminal open.

### 4. Create backend public URL

Open a third terminal from the project root:

```bash
./share_backend_cloudflared.sh
```

Copy the generated URL, for example:

```text
https://abc-backend.trycloudflare.com
```

### 5. Create frontend public URL

Open a fourth terminal from the project root:

```bash
./share_frontend_cloudflared.sh
```

Copy the generated frontend URL and open it from another machine/browser.

### 6. Configure the frontend UI

In the React page, paste the backend tunnel URL into the `Backend URL` field.

Then run:

```text
Generate Transaction
Trigger Report Generation
```

## Alternative: build with backend URL baked in

If you do not want to paste the backend URL manually in the UI, build the frontend with `VITE_API_BASE_URL`:

```bash
cd frontend
VITE_API_BASE_URL=https://your-backend-tunnel.trycloudflare.com npm run build
npm run preview:public
```

Then expose the preview server:

```bash
cloudflared tunnel --url http://127.0.0.1:4173
```

## Why Gradio is not recommended here

Gradio `share=True` is excellent when Gradio itself owns the UI. Here the UI is a Vite React app, and it calls FastAPI from the user's browser. Wrapping React inside Gradio would still require the backend to be publicly reachable. Cloudflare Tunnel or ngrok is the cleaner approach.

## Security note

Temporary tunnel URLs expose your app to anyone with the link. Do not share a tunnel while your `.env` contains real credentials or while private data is visible in the UI. Stop the tunnel terminal to revoke the temporary URL.
