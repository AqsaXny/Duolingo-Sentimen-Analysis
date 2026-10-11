# DuolingoLens — Sentiment Analytics Dashboard

> **IBM AI Builders Challenge · IT Track**  
> A production-grade sentiment analytics dashboard for Duolingo Google Play Store reviews,  
> powered by a fine-tuned **IndoBERT** model running entirely locally on Flask.

---

## Project Structure

```
DuolingoLens/
├── backend/
│   ├── app.py                      # Flask application — all API endpoints
│   └── model/                      # Downloaded model files (not committed)
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Single-page dashboard (HTML + Tailwind + Chart.js)
├── static/                         # Static assets (CSS overrides if any)
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py             # Indonesian text cleaning pipeline
│   └── slang_dict.json             # 100+ slang normalization mappings
└── uploads/                        # Temporary CSV uploads (auto-cleared)
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install flask torch transformers safetensors pandas
```

### 2. Download and Place Model Artifacts

Model dapat diunduh melalui [Google Drive](https://drive.google.com/drive/folders/1JiN-uk7Lt27nfqHSCFYnJNG4ba422YKI?usp=sharing).
Setelah diunduh, letakkan semua file model pada folder `backend/model/`:

```
backend/model/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
```

### 3. Run the Application

Jalankan perintah berikut dari **project root**:

```bash
python backend/app.py
```

Open your browser at **http://localhost:5000**

---

## API Reference

| Method | Endpoint            | Description                          |
|--------|---------------------|--------------------------------------|
| GET    | `/`                 | Serve the SPA dashboard              |
| GET    | `/api/health`       | Model + server health status         |
| POST   | `/api/predict`      | Single-text sentiment prediction     |
| POST   | `/api/predict-csv`  | Batch CSV sentiment prediction       |

### `POST /api/predict`

**Request** (JSON):
```json
{ "text": "Aplikasi ini bagus banget tapi sering error" }
```

**Response**:
```json
{
  "result": {
    "raw_text": "Aplikasi ini bagus banget tapi sering error",
    "cleaned_text": "aplikasi ini bagus banget tapi sering error",
    "label": "Negative",
    "confidence": 0.8732,
    "scores": { "Negative": 0.8732, "Positive": 0.1268 }
  },
  "analytics": {
    "total": 1,
    "positive": 0,
    "negative": 1,
    "positive_pct": 0.0,
    "negative_pct": 100.0,
    "confidence_distribution": { "Low": 0, "Medium": 0, "High": 1 },
    "word_cloud": [...],
    "rating_sentiment": {}
  }
}
```

### `POST /api/predict-csv`

**Request** (multipart/form-data):
- `file`: `.csv` file with a column named `content`, `review`, `text`, `ulasan`, `komentar`, or `teks`
- Optional: `score` or `rating` column for rating × sentiment analysis
- Max: 1 000 rows per request

**CSV Format Example**:
```csv
content,score
"Bagus banget aplikasinya",5
"Sering error dan lambat",2
```

---

## Features

| Feature | Description |
|---|---|
| **Manual Text Analysis** | Type or paste any Indonesian review for instant prediction |
| **CSV Batch Upload** | Drag-and-drop CSV upload with automatic column detection |
| **Sentiment Distribution** | Doughnut chart showing Positive / Negative split |
| **Confidence Distribution** | Bar chart bucketing model confidence (Low / Medium / High) |
| **Rating × Sentiment Anomaly Map** | Grouped bar chart detecting star-rating vs sentiment mismatches |
| **Word Cloud** | Top-60 terms from cleaned reviews, color-coded by frequency |
| **Review Table** | Paginated, filterable, searchable result table with CSV export |
| **Health Indicator** | Live model readiness badge in the navbar |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Back-End | Python · Flask 3.0 |
| ML Inference | Hugging Face Transformers · PyTorch (CPU) · safetensors |
| Preprocessing | Custom pipeline · Sastrawi (optional stemming) · Slang dict |
| Front-End | Vanilla HTML/JS · Tailwind CSS (CDN) · Chart.js 4 |
| Data I/O | pandas · Python csv / io |

---

## Preprocessing Pipeline

1. **Lowercase** all text  
2. **Remove** URLs, @mentions, #hashtags, punctuation, digits  
3. **Normalize repeated characters** (`"bagussss"` → `"baguss"`)  
4. **Slang normalization** — 100+ Indonesian internet slang mappings  
5. **Optional Sastrawi stemming** (disabled by default for latency)

---

## Model Details

- Base: `indobenchmark/indobert-base-p1` (fine-tuned)
- Task: Binary Sequence Classification (Positive / Negative)
- Max token length: 128
- Inference device: CPU (no GPU required)
- Weights format: SafeTensors

---

*Built for IBM AI Builders Challenge · IT Track · 2026*
