# ChurnGuard frontend

The frontend is a React + TypeScript single-page application built with Vite. React Router provides Overview, Customer Scoring, Analytics, Explainability, and Methodology pages. TanStack Query caches health and inference requests; Recharts visualizes only results returned for the currently loaded batch. The design uses local CSS tokens and responsive components; there is no frontend model or prediction implementation.

## Local setup

From `frontend/`:

```powershell
npm install
Copy-Item .env.example .env
npm run dev
```

Set `VITE_API_URL` to the FastAPI origin (default `http://localhost:8000`). During local Vite development, the default API address is routed through the Vite `/api` proxy to loopback, keeping the browser on one origin. Do not commit `.env`. Start the API separately from the repository root with the frozen inference artifacts available:

```powershell
.venv\Scripts\uvicorn.exe churnguard.api:app --app-dir src --reload
```

The API allows the local Vite origins on port 5173 for development. For a later hosted deployment, configure a same-origin reverse proxy or set `CHURNGUARD_CORS_ORIGINS` on the API to the exact trusted frontend origin(s), comma-separated. No production URL or credentials are embedded in this project.

## API and pages

The typed client calls `GET /health`, `POST /predict`, and `POST /batch_score`, using the current `FeatureRecord` and `ScoreResponse` contract from `src/churnguard/api.py`. The feature fields and country categories reflect the ignored `models/metadata.json` artifact present when this frontend schema was created. If inference artifacts change, update the frontend schema and category list to match the new artifact metadata; the API remains authoritative and validates every request.

Single scoring submits the current point-in-time feature snapshot (the browser derives only the deterministic log features and snapshot month already used by the model). Batch scoring validates CSV headers, values, dates, and categories in the browser, previews validation status, then submits the records through the API. Analytics are limited to returned batch results in the current app session. The Explainability page uses the top-ten mean absolute SHAP values copied from `artifacts/shap/global_feature_importance.csv`; these values are global offline contributions from the existing June feature analysis, not live customer attributions. Methodology explains the evaluation and known limitations.

## Deployment and product layers

The frontend is static and can be built with `npm run build` and hosted on Vercel or another static host once `VITE_API_URL` and browser access to the API are configured. Model files and raw datasets are not bundled with the frontend. React provides interactive scoring; FastAPI owns validation and inference; Power BI remains the separate business reporting layer.

## Tests

Run `npm test` for API request construction/error handling, response validation, environment configuration, exact feature validation, CSV validation, formatting, and an overview shell test. Run `npm run build` for TypeScript and production bundle checks.
