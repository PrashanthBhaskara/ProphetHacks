# Deploying the forecast server to Render

This deploys the FastAPI app in `server/` as a Render Web Service using
the `render.yaml` Blueprint at the repo root. Total wall time: ~10
minutes from a fresh Render account.

## Why Web Service (not the other tiles)

The screenshot Victor saw had eight choices. The one we want is
**Web Service**. Why not the others:

- **Static Site** — only serves static assets; we need a live server.
- **Private Service** — only reachable from inside Render's network;
  judges can't poll it.
- **Background Worker / Cron Job** — no inbound HTTP at all. Right
  shape if *we* were polling Kalshi, wrong shape because judges poll us.
- **Postgres / Key Value** — datastores, not compute.
- **Workflow** — batch parallel tasks (beta), not a long-running API.

## Why the Starter plan, not Free

Free Web Services spin down after 15 min of inactivity, then take
30-60 s to wake on the next request. That cold-start tax shows up as
a timeout in the judges' polling client. Starter ($7/mo) stays warm 24/7.

Total: ~$3.50 prorated for the 2-week eval window.

## Step-by-step

1. **Commit and push** the new files.

   ```bash
   git add render.yaml server/ DEPLOY.md
   git commit -m "feat(server): forecast API + Render blueprint"
   git push -u origin <branch>
   ```

   The Blueprint targets `branch: main`. If you're deploying from a
   different branch, edit `render.yaml` first or merge to main.

2. **Sign in** at [dashboard.render.com](https://dashboard.render.com)
   and connect this GitHub repo if you haven't already.
   (Account -> GitHub -> Configure -> grant access to the Prophet Hacks repo.)

3. **Apply the Blueprint.** From the dashboard:
   - New (top right) -> **Blueprint**
   - Pick the repo and the branch
   - Render reads `render.yaml`, shows a preview ("prophet-hacks-forecast,
     Starter, Oregon"), and clicks **Apply**.
   - First build takes 3-5 minutes (pip install).

4. **Note the public URL.** Render assigns
   `https://prophet-hacks-forecast.onrender.com` (or a suffix if the
   name is taken). Send this URL to whoever is polling.

5. **Verify it's live.**

   ```bash
   curl https://<your-service>.onrender.com/healthz
   # {"ok": true, "version": "1.0.0", "calibration": {"mode": "passthrough"}, ...}

   curl 'https://<your-service>.onrender.com/predict?ticker=KXATPMATCH-25JUL02SHEHIJ'
   ```

6. **Watch the logs** during the first hour of polling. Dashboard ->
   the service -> Logs. You're looking for:
   - `predict ticker=... p_yes=...` lines (normal traffic).
   - No repeating 5xx or `ConnectionError` from Kalshi (transient is fine,
     the client retries).

## Updating predictions

Any push to the configured branch redeploys automatically. To swap
the model from market mid-price to market + global Platt shrinkage,
drop a `server/calibration.json` like

```json
{"a": 1.0, "b": 0.0}
```

(use the slope/intercept your fitting script produces) and push.
The next deploy picks it up and `/healthz` will report
`"calibration": {"mode": "platt", "a": ..., "b": ...}`.

## If something goes wrong

- **Build fails on pip install** -> check `server/requirements.txt`
  versions are pinned and the `PYTHON_VERSION` env var matches a
  version Render supports (3.11.9 is current).
- **Service crashes on boot** -> the start command imports
  `server.main:app`. The Blueprint runs uvicorn from the repo root,
  so `server/__init__.py` must exist (it does).
- **Cold starts despite Starter plan** -> Render only spins down Free.
  If you accidentally selected Free in the dashboard, change it in
  Settings -> Instance Type.
- **Judges report 404 on a ticker** -> the server returns
  `p_yes: 0.5` with `source: "unknown_ticker"` instead of 404. That's
  intentional — never crash a poll. If their ticker really doesn't
  exist on Kalshi, the 0.5 fallback is the best we can do.

## Tearing down after the eval

Dashboard -> service -> Settings -> Delete Service. Or run
`render services delete <id>` if you've installed the Render CLI.
