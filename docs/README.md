# Empathy Bot Web UI

A static front-end for interacting with Empathy-Bot via a JSON API.

## Default Endpoint

Configured in `config.js`:
```
https://empathy-bot.example.com/api/chat
```

You can override it from the in-app Settings (⚙️).

## Expected Backend Contract

Request:
```json
{ "message": "User text..." }
```

Response:
```json
{ "reply": "Empathetic response here." }
```

If your backend responds with a different key (e.g. `response`), adjust `queryBackend` in `script.js`.

## Features

- Local conversation persistence (localStorage)
- Light/Dark theme toggle
- Endpoint override dialog
- Optional simulated streaming effect
- Minimal markdown (bold, italics, inline code)
- Accessible semantics (landmarks, ARIA, keyboard focus)
- Graceful error display

## Deployment (GitHub Pages via Actions)

1. Make sure Pages is set to “GitHub Actions” under Settings > Pages.
2. The workflow `.github/workflows/deploy-pages.yml` deploys changes when files under `docs/` (or the workflow itself) update on `main`.

Site URL pattern:
```
https://<your-username>.github.io/Empathy-Bot/
```

## Customization Tips

- Real streaming: Replace `simulateStreaming` with a reader loop using `fetch` + `ReadableStream`.
- Rich markdown: Swap `renderMarkdownBasic` with a library (e.g. marked).
- Authentication: Use a backend proxy; do not embed secrets in the client.

## Security Notes

- No API keys or secrets should appear in this repository.
- Implement rate-limiting or abuse protection server-side.

---

Licensed under the root project’s license.