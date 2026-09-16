# Deployment Guide for WhyWrong

Because WhyWrong is distilled down to INT8 ONNX and precomputed lookups, it runs on **1 CPU core and requires under 150 MB of RAM**. It needs no GPU and no external API keys, making it 100% free to host.

---

## Step 1: Push to GitHub

1. Create a new repository on your GitHub account ([github.com/new](https://github.com/new)):
   - Name: `whywrong` (or any name you prefer)
   - Set to **Public**
   - Do **NOT** check "Add a README" or ".gitignore" (these already exist in the project).

2. Link your local project to your new repository and push:
   ```bash
   git remote add origin https://github.com/bluebox123/whywrong.git
   git branch -M main
   git push -u origin main
   ```

---

## Step 2: Deploy to a Free Cloud Host (Choose Option A or B)

### Option A: Render (Easiest — 1 Click with GitHub)
1. Go to [render.com](https://render.com) and sign in with your GitHub account.
2. Click **New +** &rarr; **Blueprint** (or **Web Service**).
3. Connect your `whywrong` repository.
4. Render will automatically detect `render.yaml` and `Dockerfile`.
5. Select the **Free** plan and click **Apply** / **Create Web Service**.
6. In ~2 minutes, your live demo URL will be active (e.g., `https://whywrong-xxxx.onrender.com`).

### Option B: Hugging Face Spaces (Recommended for ML Demos)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Set Space name (e.g. `whywrong-demo`), license: `mit`.
3. Choose **Docker** as the Space SDK (Blank).
4. Select the **Free** hardware tier (2 vCPU, 16GB RAM).
5. Follow the instructions to push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/whywrong-demo
   git push hf main
   ```
6. Your Space will build the Docker container and be live permanently at:
   `https://YOUR_HF_USERNAME-whywrong-demo.hf.space`

---

## Step 3: Instant Live URL (Alternative: Cloudflare Tunnel)
If you need an immediate public HTTPS URL for recording or previewing right now without waiting for cloud builds:
```bash
# In PowerShell / Command Prompt:
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://127.0.0.1:8077
```
Cloudflare will immediately generate a public HTTPS URL (e.g., `https://xxxx.trycloudflare.com`) routing directly to your running local server!
