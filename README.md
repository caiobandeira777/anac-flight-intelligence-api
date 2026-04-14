# ✈️ ANAC Flight Intelligence API

A production-ready machine learning API that predicts airport congestion, seat occupancy, ticket pricing pressure, and baggage excess probability for Brazilian aviation routes — built on 20GB+ of open ANAC (Brazilian Civil Aviation Agency) data.

## 🚀 Live Demo
API running at: `http://localhost:8001/docs`

## 🧠 Models

| Model | Architecture | Task | Key Metric |
|-------|-------------|------|------------|
| Airport Congestion | Bidirectional LSTM | Regression | MAE on hourly flow |
| Seat Occupancy | FT-Transformer | Multi-task (reg + clf) | MAE + AUC |
| Price Pressure | MLP (PyTorch) | Regression | MAE on price index |
| Baggage Excess | LightGBM + DNN | Binary classification | AUC-ROC |

## 📦 Tech Stack

- **ML:** PyTorch, LightGBM, scikit-learn, HuggingFace-style Transformer
- **API:** FastAPI + Uvicorn
- **Data:** Polars (20GB+ Parquet pipeline)
- **MLOps:** MLflow experiment tracking, drift monitoring (PSI + MAE threshold alerts)
- **Infra:** Docker, Railway-ready

## 🗂️ Project Structure

```
projeto-anac/
│
├── 01_ingestao.py                 # Raw CSV → Parquet ingestion (handles 20GB+)
├── 02_feature_engineering.py      # Feature pipelines for all 4 models
├── 03_modelo_aeroporto.py         # Bidirectional LSTM — airport congestion
├── 04_modelo_assentos.py          # FT-Transformer — seat occupancy (multi-task)
├── 05_modelos_preco_bagagem.py    # MLP (pricing) + LightGBM+DNN stack (baggage)
├── 06_api.py                      # FastAPI inference server
├── 07_monitorar_drift.py          # Monthly drift monitoring for all 4 models
│
├── data/
│   ├── raw/                       # Original ANAC CSV files (~20GB)
│   ├── processed/                 # Intermediate Parquet
│   ├── features/                  # Model-ready feature tables
│   │   ├── feat_aeroporto.parquet
│   │   ├── feat_assentos.parquet
│   │   ├── feat_precificacao.parquet
│   │   └── feat_bagagem.parquet
│   └── drift/                     # Drift monitoring reports (JSON)
│
├── models/
│   ├── modelo_aeroporto.pt        # LSTM checkpoint
│   ├── modelo_assentos.pt         # FT-Transformer checkpoint
│   ├── modelo_precificacao.pt     # MLP checkpoint
│   ├── lgb_bagagem.txt            # LightGBM booster
│   └── dnn_bagagem.pt             # DNN stacking layer checkpoint
│
├── dashboard.html                 # Static analytics dashboard
├── Dockerfile                     # Production image (Python 3.11-slim + CPU PyTorch)
├── docker-compose.yml             # API + Nginx dashboard in one command
└── requirements.txt
```

## ⚡ Quick Start

### Docker (recommended)

```bash
docker-compose up --build
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:3000`

### Local

```bash
pip install -r requirements.txt

# Run the full pipeline (only needed once)
python 01_ingestao.py
python 02_feature_engineering.py
python 03_modelo_aeroporto.py
python 04_modelo_assentos.py
python 05_modelos_preco_bagagem.py

# Start the API
uvicorn 06_api:app --reload --port 8000
```

## 🔌 API Usage

**Single endpoint — all predictions in one call:**

```bash
curl -X POST http://localhost:8001/prever \
  -H "Content-Type: application/json" \
  -d '{
    "aeroporto_origem": "GRU",
    "aeroporto_destino": "GIG",
    "empresa": "LATAM",
    "data_voo": "2026-07-15",
    "peso_bagagem_kg": 20.0
  }'
```

**Response:**
```json
{
  "prob_aeroporto_cheio": 0.74,
  "taxa_ocupacao_voo": 0.89,
  "bucket_ocupacao": "lotado",
  "pressao_preco": 0.638,
  "recomendacao_preco": "premium",
  "prob_excesso_bagagem": 0.038,
  "recomendar_cobranca": false,
  "justificativa_bagagem": "Baixo histórico de excesso nesta rota"
}
```

## ⚙️ Running Locally

**With Docker (recommended):**
```bash
docker build -t anac-api .
docker run -p 8001:8001 anac-api
```

**Without Docker:**
```bash
pip install -r requirements.txt
uvicorn 06_api:app --reload --port 8001
```

## 📊 Data Pipeline

- Source: [ANAC Open Data](https://www.anac.gov.br/acesso-a-informacao/dados-abertos)
- Raw data: ~20GB CSV files (domestic + international flights 2018–2024)
- Processed: Parquet format via Polars for high-performance feature engineering
- Features: route statistics, seasonality, carrier history, airport flow patterns

## 🔍 MLOps & Monitoring

- **Drift detection:** PSI (Population Stability Index) + MAE threshold alerts for all 4 models
- **Reproducibility:** Fixed seeds (42) across all training scripts
- **Experiment tracking:** MLflow integration
- **Cache layer:** Built-in prediction caching with TTL

## 📈 Key Engineering Decisions

- **FT-Transformer for tabular data:** Outperformed XGBoost on seat occupancy due to complex feature interactions
- **Multi-task learning:** Joint seat occupancy + overbook probability sharing representations
- **Stratified splits + scale_pos_weight:** Handles class imbalance in baggage excess prediction
- **Route-aware inference:** Historical stats lookup by route+carrier with 3-level fallback

## 🛠️ Author

**Caio Bandeira**
- Data Science & ML student @ UniCEUB
- Former Parliamentary Data Analyst @ Câmara dos Deputados
- [LinkedIn](https://linkedin.com/in/bandeira-caio) | [GitHub](https://github.com/caiobandeira777)

---
*Data source: ANAC (Agência Nacional de Aviação Civil) — publicly available under Brazilian open data law*
