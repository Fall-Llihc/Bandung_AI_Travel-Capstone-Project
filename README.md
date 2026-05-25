# Bandung AI Travel — Capstone Project

> **Smart trip planner untuk Bandung Raya.** User isi profil singkat (kategori suka, budget, durasi, titik berangkat) → sistem rekomendasikan 4–6 destinasi yang relevan, plus narasi cerita perjalanan yang dihasilkan LLM.

Stack: **React (Vercel)** + **FastAPI (Railway)** + **CBF + Q-Learning RL** untuk rekomendasi + **Groq Llama-3.1** untuk storyteller.

| Layer    | URL Production                                                                  | Tech                                  |
| -------- | ------------------------------------------------------------------------------- | ------------------------------------- |
| Frontend | _(deploy via Vercel — rujuk `frontend/.env.production`)_                        | React 18, react-scripts 5             |
| Backend  | `https://bandungaitravel-capstone-project-production.up.railway.app/api/health` | FastAPI, scikit-learn, Groq SDK       |

---

## Arsitektur singkat

```
┌──────────────┐   POST /api/plan            ┌────────────────────────┐
│              │ ─────────────────────────▶ │  FastAPI (Railway)     │
│  React UI    │                            │  ┌─────────────────┐   │
│  (Vercel)    │                            │  │ Recommender     │   │
│              │ ◀───────────────────────── │  │  • CBF          │   │
└──────────────┘   itinerary + narrative    │  │  • Q-Learning   │   │
                                            │  │  • category-first│  │
                                            │  └─────────────────┘   │
                                            │  ┌─────────────────┐   │
                                            │  │ LLM Storyteller │   │
                                            │  │  (Groq)         │   │
                                            │  └─────────────────┘   │
                                            └────────────────────────┘
```

Pipeline `/api/plan`:

1. Frontend kirim profil user (`categories`, `budget`, `home`, opsional `duration_hours`, `radius_km`, `top_n`).
2. **Recommender** (`backend/recommender.py`):
   - **Filter** dataset berdasarkan kategori + radius dari titik berangkat (jarak via formula Haversine).
   - **CBF (Content-Based Filtering)**: ranking awal pakai cosine-similarity antar feature vector destinasi (one-hot kategori + numeric scaled + TF-IDF deskripsi). Reservasi slot per kategori dilakukan **sebelum** RL fill — supaya kategori yang user pilih dijamin terwakili.
   - **DRL re-ranker (Q-Learning)**: pilih action dari kandidat top sambil mempertimbangkan reward (rating, jarak, budget gate, variety bonus). Mengisi slot sisa setelah reservasi kategori.
3. **Storyteller** (`backend/llm_storyteller.py`): generate narasi 1–2 paragraf via Groq + retry/fallback.
4. Response final: list destinasi terurut + `narrative` + metadata (`total_km`, `total_cost`, dll).

Sample request/response: lihat [`docs/api/sample_request.json`](docs/api/sample_request.json) & [`docs/api/sample_response.json`](docs/api/sample_response.json).

---

## Dataset & model artifacts

Snapshot terkini (sudah committed di repo):

| Item                                | Nilai                                       |
| ----------------------------------- | ------------------------------------------- |
| `backend/data/destinations.csv`     | **1.459 destinasi** Bandung Raya            |
| Distribusi kategori                 | Alam (722) · Kuliner (645) · Wisata (92)    |
| `backend/data/last_updated.txt`     | `2026-05-25`                                |
| `feature_matrix` (CBF input)        | shape `(1459, 28)`, dtype `float64`         |
| Komposisi feature                   | 3 one-hot kategori (×2.0) + 5 numeric scaled (×1.0) + up to 20 TF-IDF (×0.5) |
| `backend/models/cbf_model.pkl`      | `similarity_matrix` (1459×1459) + `df_index`        |
| `backend/models/rl_agent.pkl`       | Q-table dict, dilatih `N_EPISODES = 3000`   |
| `backend/models/scaler.pkl`         | MinMaxScaler untuk fitur numerik            |
| `backend/models/label_encoders.pkl` | encoder kategori & tag                      |

Eval ringkas dari training terbaru ([`eval_report.json`](https://github.com/Fall-Llihc/Bandung_AI_Travel-Capstone-Project/blob/updateVer/HASIL_TERBARU/working/data/processed/eval_report.json) di branch `updateVer`):

| Metrik                  | Nilai     |
| ----------------------- | --------- |
| n_scenarios             | 100       |
| category_coverage_pct   | 97.0%     |
| distance_compliance_pct | 100.0%    |
| budget_compliance_pct   | 83.0%     |
| avg_rating              | 4.32      |
| avg_total_km            | 47.0 km   |
| avg_total_cost          | Rp 185.640|
| avg_variety_index       | 0.97      |
| avg_steps_per_itinerary | 3.31      |

Sumber data: scrape OSM (Overpass API) → cleaning → enrichment manual untuk kuliner ikonik. Notebook lengkap: [`notebooks/rec-engine.ipynb`](notebooks/rec-engine.ipynb).

---

## Struktur direktori

```
Bandung_AI_Travel-Capstone-Project/
├── backend/                          # FastAPI service (deploy → Railway)
│   ├── main.py                       # API entry: /, /api/health, /api/plan
│   ├── recommender.py                # CBF + Q-Learning + category-first reservation
│   ├── llm_storyteller.py            # Groq LLM wrapper (retry + fallback)
│   ├── data/
│   │   ├── destinations.csv          # dataset runtime
│   │   └── last_updated.txt
│   ├── models/                       # pkl artefak runtime
│   │   ├── cbf_model.pkl
│   │   ├── rl_agent.pkl
│   │   ├── scaler.pkl
│   │   └── label_encoders.pkl
│   ├── requirements.txt
│   ├── runtime.txt                   # Python 3.11
│   ├── Procfile                      # uvicorn main:app ...
│   ├── railway.json
│   └── .env.example                  # GROQ_API_KEY, GROQ_MODEL, ALLOWED_ORIGINS, ...
│
├── frontend/                         # React SPA (deploy → Vercel)
│   ├── src/
│   │   ├── App.jsx                   # state machine: welcome → form → loading → results
│   │   ├── components/               # WelcomeScreen, FormScreen, LoadingScreen, ResultsScreen
│   │   ├── api/client.js             # fetch wrapper ke /api/plan
│   │   ├── data/homeOptions.js       # 5 preset titik berangkat
│   │   └── utils/format.js
│   ├── public/index.html
│   ├── package.json
│   ├── vercel.json
│   ├── .env.development              # REACT_APP_API_BASE_URL=http://localhost:8000
│   └── .env.production               # REACT_APP_API_BASE_URL=<railway url>
│
├── notebooks/
│   ├── rec-engine.ipynb              # versi yang dipakai untuk train (Kaggle-friendly)
│   ├── rec-engine (4).ipynb          # arsip eksperimen
│   ├── llm-train.ipynb               # eksperimen prompt LLM
│   └── _build_notebook.py            # script generator notebook dari .py source
│
├── models/                           # mirror artefak (legacy, dibiarkan untuk konsistensi tooling)
│
├── docs/
│   ├── api/
│   │   ├── sample_request.json
│   │   └── sample_response.json
│   └── screenshots/                  # bukti UI/UX (welcome, form, results)
│
├── scripts/
│   └── apply-kaggle-artifacts.sh     # apply zip dari Kaggle ke layout backend/
│
├── requirements.txt                  # untuk training/notebook (numpy, pandas, sklearn, dll)
└── README.md
```

---

## Quickstart — local development

### Prasyarat

* Python 3.11
* Node 18+ (sesuai `frontend/package.json`)
* (Opsional) Groq API key — kalau kosong, storyteller jatuh ke template fallback.

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # isi GROQ_API_KEY kalau punya
uvicorn main:app --reload --port 8000
```

Smoke test:

```bash
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/plan \
     -H "Content-Type: application/json" \
     --data @docs/api/sample_request.json
```

### Frontend

```bash
cd frontend
npm install
npm start          # buka http://localhost:3000
```

Defaultnya frontend akan hit `http://localhost:8000` (lihat `.env.development`). Untuk pakai backend production, override via `.env.local` atau set `REACT_APP_API_BASE_URL`.

---

## API reference

| Method | Path          | Deskripsi                                                   |
| ------ | ------------- | ----------------------------------------------------------- |
| GET    | `/`           | Service info (nama, versi, daftar endpoint)                 |
| GET    | `/api/health` | Health check + `last_updated`, `n_destinations`, status LLM |
| POST   | `/api/plan`   | Generate itinerary + narasi                                 |

### `POST /api/plan` — request

```json
{
  "categories": ["Alam", "Kuliner", "Wisata"],
  "budget": 300000,
  "home": "Stasiun Bandung",
  "duration_hours": 8,
  "radius_km": 25,
  "top_n": 6
}
```

* `categories`: subset dari `["Alam", "Kuliner", "Wisata"]`
* `home`: salah satu preset (`Stasiun Bandung`, `Dago`, `Lembang`, `Soreang`, `Cimahi`) — koordinat dipetakan di `frontend/src/data/homeOptions.js`
* `radius_km` & `duration_hours` opsional, dipakai sebagai hard-gate filter
* `top_n` opsional, default 6

### `POST /api/plan` — response (singkat)

```json
{
  "items": [
    {
      "id": "...",
      "name": "...",
      "category": "Alam",
      "lat": -6.825, "lng": 107.601,
      "ticket": 25000, "duration": 90, "rating": 4.6,
      "stay_detail": "...", "gmaps_url": "..."
    }
  ],
  "narrative": "Hari ini perjalananmu dimulai dari ...",
  "summary": {
    "total_cost": 175000,
    "total_km": 41.2,
    "total_minutes": 380,
    "n_items": 5
  },
  "meta": { "last_updated": "2026-05-25", "model_version": "..." }
}
```

Full sample: [`docs/api/sample_response.json`](docs/api/sample_response.json).

---

## Mengupdate model & dataset (training → production)

Ada **dua cara** sesuai workflow tim:

### Opsi A — via branch `updateVer/HASIL_TERBARU/working/` (preferred)

Notebook Kaggle commit hasil training ke branch `updateVer` di path `HASIL_TERBARU/working/`. Untuk apply ke `main`:

```bash
git fetch origin updateVer
git checkout -b chore/update-artifacts-YYYY-MM-DD origin/main

# Salin artefak dari updateVer ke layout deployable
git cat-file -p updateVer:HASIL_TERBARU/working/data/processed/destinations.csv > backend/data/destinations.csv
git cat-file -p updateVer:HASIL_TERBARU/working/data/last_updated.txt           > backend/data/last_updated.txt
for f in cbf_model rl_agent scaler label_encoders; do
  git cat-file -p updateVer:HASIL_TERBARU/working/models/${f}.pkl > backend/models/${f}.pkl
  git cat-file -p updateVer:HASIL_TERBARU/working/models/${f}.pkl > models/${f}.pkl
done
git cat-file -p updateVer:HASIL_TERBARU/working/data/processed/sample_api_request.json  > docs/api/sample_request.json
git cat-file -p updateVer:HASIL_TERBARU/working/data/processed/sample_api_response.json > docs/api/sample_response.json

git add backend/data backend/models docs/api models
git commit -m "chore: update artifacts from updateVer training (YYYY-MM-DD)"
```

Contoh nyata: [PR #10](https://github.com/Fall-Llihc/Bandung_AI_Travel-Capstone-Project/pull/10).

### Opsi B — via Kaggle artifact ZIP (lama)

Kalau Kaggle output di-export sebagai `bandung-travel-artifacts.zip`:

```bash
./scripts/apply-kaggle-artifacts.sh /path/to/bandung-travel-artifacts.zip
```

Script ini extract zip lalu menyalin pkl + csv ke `backend/models/`, `backend/data/`, dan sample API ke `docs/api/`.

### Setelah commit & merge

* **Railway** otomatis redeploy backend → loader baca `backend/models/*.pkl` + `backend/data/*` saat startup.
* **Vercel** _tidak perlu_ redeploy kecuali `frontend/` ikut berubah.
* Verifikasi: `curl https://<railway-url>/api/health` — pastikan `last_updated` & `n_destinations` sesuai snapshot baru.

---

## Catatan implementasi

### Recommender — kategori-first reservation

`Recommender.recommend()` melakukan reservasi slot per kategori (proporsional ke `top_n`) **sebelum** Q-Learning fill. Tujuannya: hindari kasus dimana DRL terlalu "rakus" pada kategori dengan reward tertinggi sehingga kategori lain yang user pilih jadi kosong. Implementasi di method `_reserve_per_category` (lihat fix di [PR #9](https://github.com/Fall-Llihc/Bandung_AI_Travel-Capstone-Project/pull/9)).

### Hard-gate constraints

* **Radius**: filter Haversine dari `home` sebelum scoring.
* **Budget**: penalti besar di reward RL kalau total ticket > budget; UI juga warn user.
* **Duration**: total `duration` + estimasi travel time (`CITY_TRAVEL_SPEED_KMH`) ≤ `duration_hours`.

### CBF kompatibilitas dua format

`recommender.py` mendukung dua skema pickle agar tahan terhadap berbagai versi notebook:

* Format baru (Kaggle): `{"similarity_matrix": np.ndarray, "df_index": [{...}]}`
* Format lama: `{"sim_matrix": ..., "id_to_sim_idx": {...}}`

Loader otomatis fallback — tidak perlu code change saat ganti pkl.

### LLM storyteller

* Default model: `llama-3.1-8b-instant` (override via `GROQ_MODEL` env).
* Cold-start timeout 12 detik. Kalau Groq lambat / quota habis / API key kosong → fallback template deterministik berbasis nama destinasi & kategori.
* Free tier Groq cukup besar (~14.400 request/hari) untuk demo.

---

## Lihat juga

* Sample I/O: [`docs/api/`](docs/api/)
* Screenshot UI: [`docs/screenshots/`](docs/screenshots/)
* Notebook training: [`notebooks/rec-engine.ipynb`](notebooks/rec-engine.ipynb)
* Notebook eksperimen LLM: [`notebooks/llm-train.ipynb`](notebooks/llm-train.ipynb)

---

## Lisensi

Capstone project — internal academic use. Lihat repo owner ([Fall-Llihc](https://github.com/Fall-Llihc)) untuk pertanyaan lisensi.
