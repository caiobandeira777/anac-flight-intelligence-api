"""
=============================================================
  MONITORAMENTO DE DRIFT
  Compara as predições históricas com os dados reais da ANAC
  e avisa quando o modelo começa a ficar desatualizado.
  
  Rodar: python 07_monitorar_drift.py
  Agendar: executar mensalmente após novos dados da ANAC
=============================================================
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("models")
DRIFT_DIR    = Path("data/drift")
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

# limiares de alerta — se o erro ultrapassar, emite aviso
LIMIARES = {
    "mae_ocupacao":        0.12,   # MAE de taxa_ocupacao > 12% = drift (assentos)
    "auc_bagagem":         0.80,   # AUC abaixo de 80% = drift (bagagem)
    "mae_pressao":         0.15,   # MAE de pressao_preco > 15% = drift (precificação)
    "mae_aeroporto":       0.20,   # MAE normalizado de decolagens > 20% = drift (aeroporto)
}


def avaliar_modelo_assentos() -> dict:
    """
    Compara a taxa_ocupacao real (nos dados) com o que o modelo preveria.
    Usa os dados de 2025 como 'janela de monitoramento' — dados que o modelo
    nunca viu durante o treino (treino foi até 2023).
    """
    log.info("Avaliando modelo de assentos (FT-Transformer)...")

    import torch
    from sklearn.preprocessing import StandardScaler

    df = pl.read_parquet(FEATURES_DIR / "feat_assentos.parquet")

    # janela de monitoramento: dados mais recentes
    df_monitor = df.filter(pl.col("nr_ano_referencia").cast(pl.Int32) >= 2025)

    if len(df_monitor) == 0:
        log.warning("  Sem dados de 2025 para monitorar assentos.")
        return {"status": "sem_dados", "mae": None}

    # amostra para avaliar (máx 50k linhas para ser rápido)
    df_monitor = df_monitor.sample(n=min(50_000, len(df_monitor)), seed=42)

    # carrega o modelo
    ckpt = torch.load(MODELS_DIR / "modelo_assentos.pt", map_location="cpu", weights_only=False)

    from torch import nn

    class FeatureTokenizer(nn.Module):
        def __init__(self, vocab_sizes, n_num, embed_dim):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes])
            self.num_proj = nn.Linear(1, embed_dim)
            self.n_num = n_num
        def forward(self, x_cat, x_num):
            tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            for i in range(self.n_num): tokens.append(self.num_proj(x_num[:, i].unsqueeze(1)))
            return torch.stack(tokens, dim=1)

    class FTTransformer(nn.Module):
        def __init__(self, vocab_sizes, n_num, cfg):
            super().__init__()
            self.tokenizer = FeatureTokenizer(vocab_sizes, n_num, cfg["embed_dim"])
            self.cls_token = nn.Parameter(torch.randn(1, 1, cfg["embed_dim"]))
            enc = nn.TransformerEncoderLayer(d_model=cfg["embed_dim"], nhead=cfg["n_heads"], dim_feedforward=cfg["ffn_dim"], dropout=cfg["dropout"], batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(enc, num_layers=cfg["n_layers"])
            self.cabeca_reg = nn.Sequential(nn.Linear(cfg["embed_dim"], 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            self.cabeca_clf = nn.Sequential(nn.Linear(cfg["embed_dim"], 64), nn.ReLU(), nn.Linear(64, 4))
        def forward(self, x_cat, x_num):
            tokens = self.tokenizer(x_cat, x_num)
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            out = self.transformer(torch.cat([cls, tokens], dim=1))
            cls_out = out[:, 0, :]
            return self.cabeca_reg(cls_out), self.cabeca_clf(cls_out)

    modelo = FTTransformer(ckpt["vocab_sizes"], ckpt["n_numericas"], ckpt["config"])
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()

    # encoda features
    from sklearn.preprocessing import LabelEncoder
    cats = ckpt["categoricas"]
    encoders = {k: type('E', (), {'classes_': v, 'transform': lambda self, x: [list(self.classes_).index(i) if i in self.classes_ else 0 for i in x]})() for k, v in ckpt["encoders"].items()}

    cat_arrays = []
    for col in cats:
        vals = df_monitor[col].fill_null("__desconhecido__").to_numpy().astype(str)
        lookup = {v: i for i, v in enumerate(ckpt["encoders"][col])}
        cat_arrays.append(np.array([lookup.get(v, 0) for v in vals], dtype=np.int64))

    x_cat = torch.tensor(np.array(cat_arrays, dtype=np.int64).T, dtype=torch.long)

    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])
    num_np = df_monitor.select(ckpt["numericas"]).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
    x_num  = torch.tensor(sc.transform(num_np), dtype=torch.float32)
    y_real = df_monitor["taxa_ocupacao"].fill_null(0.0).to_numpy()

    # prediz em batches
    preds = []
    batch = 1024
    for i in range(0, len(x_cat), batch):
        with torch.no_grad():
            pred, _ = modelo(x_cat[i:i+batch], x_num[i:i+batch])
            preds.append(pred.numpy().flatten())
    y_pred = np.concatenate(preds)

    mae    = float(np.mean(np.abs(y_real - y_pred)))
    status = "🔴 DRIFT DETECTADO" if mae > LIMIARES["mae_ocupacao"] else "🟢 OK"

    log.info(f"  MAE ocupação (2025): {mae:.4f} | limiar: {LIMIARES['mae_ocupacao']} | {status}")
    return {"status": status, "mae": round(mae, 4), "n_amostras": len(df_monitor), "limiar": LIMIARES["mae_ocupacao"]}


def avaliar_modelo_bagagem() -> dict:
    """Avalia o LightGBM de bagagem nos dados mais recentes."""
    log.info("Avaliando modelo de bagagem (LightGBM)...")

    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    df = pl.read_parquet(FEATURES_DIR / "feat_bagagem.parquet")
    df_monitor = df.filter(pl.col("nr_mes_referencia").cast(pl.Int32).is_in([11, 12]))  # últimos meses

    CAT_BAGAGEM = ["rota_od","sg_empresa_icao","nm_regiao_origem","nm_regiao_destino",
                   "nm_continente_destino","faixa_distancia","ds_tipo_linha","nm_dia_semana_referencia"]

    for col in CAT_BAGAGEM:
        df_monitor = df_monitor.with_columns(pl.col(col).fill_null("desconhecido"))
    df_monitor = df_monitor.with_columns([
        pl.col("taxa_excesso_historica_rota").fill_null(0.0).fill_nan(0.0),
        pl.col("kg_excesso_medio_rota").fill_null(0.0).fill_nan(0.0),
        pl.col("taxa_excesso_historica_empresa").fill_null(0.0).fill_nan(0.0),
        pl.col("nr_passag_pagos").fill_null(0.0),
        pl.col("km_distancia").fill_null(0.0),
        pl.col("flag_bagagem_excesso").fill_null(0),
    ])

    FEATURES = ["rota_od","sg_empresa_icao","nm_regiao_origem","nm_regiao_destino",
                "nm_continente_destino","flag_internacional","nr_mes_referencia",
                "nm_dia_semana_referencia","nr_hora_partida_real","km_distancia",
                "faixa_distancia","ds_tipo_linha","nr_passag_pagos",
                "taxa_excesso_historica_rota","kg_excesso_medio_rota","taxa_excesso_historica_empresa"]

    X = df_monitor.select(FEATURES).to_pandas()
    for col in CAT_BAGAGEM:
        X[col] = X[col].astype("category")
    X["nr_mes_referencia"] = X["nr_mes_referencia"].astype(str).str.extract(r'(\d+)').astype(float).astype(int)
    y = df_monitor["flag_bagagem_excesso"].to_numpy().astype(int)

    if len(y) < 100 or y.sum() < 10:
        log.warning("  Dados insuficientes para avaliar bagagem.")
        return {"status": "sem_dados", "auc": None}

    modelo = lgb.Booster(model_file=str(MODELS_DIR / "lgb_bagagem.txt"))
    prob   = modelo.predict(X)
    auc    = roc_auc_score(y, prob)
    status = "🔴 DRIFT DETECTADO" if auc < LIMIARES["auc_bagagem"] else "🟢 OK"

    log.info(f"  AUC bagagem (monitor): {auc:.4f} | limiar: {LIMIARES['auc_bagagem']} | {status}")
    return {"status": status, "auc": round(auc, 4), "n_amostras": len(df_monitor), "limiar": LIMIARES["auc_bagagem"]}


def avaliar_modelo_aeroporto() -> dict:
    """
    Avalia o modelo LSTM de lotação de aeroporto nos dados de 2025.
    Reconstrói sequências de 48h, prediz as próximas 12h e compara
    o primeiro passo da predição com o valor real (total_decolagens normalizado).
    """
    log.info("Avaliando modelo de aeroporto (LSTM bidirecional)...")

    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler

    FEATURES_SERIE = [
        "total_decolagens", "total_passageiros", "total_assentos",
        "ocupacao_media", "voos_distintos", "dia_semana_num",
        "nr_hora_partida_real", "semana_ano", "flag_feriado",
    ]
    JANELA    = 48
    HORIZONTE = 12

    df = pl.read_parquet(FEATURES_DIR / "feat_aeroporto.parquet")
    df_monitor = df.filter(pl.col("dt_partida_real").dt.year() >= 2025)

    if len(df_monitor) == 0:
        log.warning("  Sem dados de 2025 para monitorar aeroporto.")
        return {"status": "sem_dados", "mae": None}

    ckpt = torch.load(MODELS_DIR / "modelo_aeroporto.pt", map_location="cpu", weights_only=False)

    # reconstrói scaler a partir do checkpoint
    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])

    # preenche nulos (mesmo pré-processamento do treino)
    df_monitor = df_monitor.with_columns([
        pl.col(c).fill_null(0).fill_nan(0) for c in FEATURES_SERIE
    ])

    erros = []
    for icao in df_monitor["sg_icao_origem"].unique().to_list():
        serie_raw = (
            df_monitor.filter(pl.col("sg_icao_origem") == icao)
            .sort(["dt_partida_real", "nr_hora_partida_real"])
            .select(FEATURES_SERIE)
            .to_numpy().astype(np.float32)
        )
        if len(serie_raw) < JANELA + HORIZONTE:
            continue
        serie = sc.transform(serie_raw)
        # avalia até 50 janelas por aeroporto para ser rápido
        for i in range(0, min(len(serie) - JANELA - HORIZONTE, 50)):
            X_np = serie[i : i + JANELA]
            y_real = serie[i + JANELA, 0]   # total_decolagens normalizado, 1º passo
            erros.append(abs(y_real))        # baseline: predição = 0 (worst case)

    # ── reconstrói o modelo ──────────────────────────────────────────────────
    class AtencaoTemporal(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.a = nn.Linear(h * 2, 1)
        def forward(self, x):
            return (x * torch.softmax(self.a(x), dim=1)).sum(dim=1)

    class LSTMAeroporto(nn.Module):
        def __init__(self, n_feat, horizonte, cfg):
            super().__init__()
            h = cfg["hidden_size"]
            self.lstm = nn.LSTM(n_feat, h, cfg["num_layers"], batch_first=True,
                                bidirectional=True,
                                dropout=cfg["dropout"] if cfg["num_layers"] > 1 else 0.0)
            self.atencao = AtencaoTemporal(h)
            self.head = nn.Sequential(nn.Linear(h * 2, 128), nn.ReLU(),
                                      nn.Dropout(cfg["dropout"]), nn.Linear(128, horizonte * 2))
            self.horizonte = horizonte
        def forward(self, x):
            out, _ = self.lstm(x)
            ctx = self.atencao(out)
            return self.head(ctx).view(-1, self.horizonte, 2)

    cfg    = ckpt["config"]
    modelo = LSTMAeroporto(len(FEATURES_SERIE), cfg["horizonte"], cfg)
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()

    # avalia com predições reais (amostra dos aeroportos monitorados)
    erros_reais = []
    n_avaliados = 0
    for icao in df_monitor["sg_icao_origem"].unique().to_list():
        serie_raw = (
            df_monitor.filter(pl.col("sg_icao_origem") == icao)
            .sort(["dt_partida_real", "nr_hora_partida_real"])
            .select(FEATURES_SERIE)
            .to_numpy().astype(np.float32)
        )
        if len(serie_raw) < JANELA + HORIZONTE:
            continue
        serie = sc.transform(serie_raw)
        for i in range(0, min(len(serie) - JANELA - HORIZONTE, 20)):
            X_t  = torch.tensor(serie[i : i + JANELA], dtype=torch.float32).unsqueeze(0)
            y_real = float(serie[i + JANELA, 0])
            with torch.no_grad():
                pred = float(modelo(X_t)[0, 0, 0])
            erros_reais.append(abs(y_real - pred))
            n_avaliados += 1

    if n_avaliados == 0:
        log.warning("  Nenhuma sequência avaliada para o modelo de aeroporto.")
        return {"status": "sem_dados", "mae": None}

    mae    = float(np.mean(erros_reais))
    status = "🔴 DRIFT DETECTADO" if mae > LIMIARES["mae_aeroporto"] else "🟢 OK"
    log.info(
        f"  MAE aeroporto normalizado (2025): {mae:.4f} | "
        f"limiar: {LIMIARES['mae_aeroporto']} | {status} | n={n_avaliados}"
    )
    return {
        "status":    status,
        "mae":       round(mae, 4),
        "n_amostras": n_avaliados,
        "limiar":    LIMIARES["mae_aeroporto"],
        "nota":      "MAE no espaço normalizado (StandardScaler do treino)",
    }


def avaliar_modelo_precificacao() -> dict:
    """
    Avalia o MLP de precificação nos dados dos meses 11 e 12 (mesma janela
    usada no treino para validação) e compara pressao_preco prevista vs real.
    """
    log.info("Avaliando modelo de precificação (MLP)...")

    import torch
    from torch import nn
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    CATEGORICAS_PRECO = ["rota_od", "sg_empresa_icao", "nm_dia_semana_referencia"]
    NUMERICAS_PRECO   = [
        "nr_mes_referencia", "nr_hora_partida_real", "km_distancia",
        "ocupacao_media", "combustivel_medio", "load_factor_medio", "frequencia_voos",
    ]

    df = pl.read_parquet(FEATURES_DIR / "feat_precificacao.parquet")
    df_monitor = df.filter(pl.col("nr_mes_referencia").cast(pl.Int32).is_in([11, 12]))

    if len(df_monitor) == 0:
        log.warning("  Sem dados de meses 11/12 para monitorar precificação.")
        return {"status": "sem_dados", "mae": None}

    df_monitor = df_monitor.sample(n=min(50_000, len(df_monitor)), seed=42)

    ckpt = torch.load(MODELS_DIR / "modelo_precificacao.pt", map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    # ── reconstrói encoders a partir do checkpoint ───────────────────────────
    cat_arrays = []
    for col in CATEGORICAS_PRECO:
        vals   = df_monitor[col].fill_null("__desconhecido__").to_numpy().astype(str)
        lookup = {v: i for i, v in enumerate(ckpt["encoders"][col])}
        cat_arrays.append(np.array([lookup.get(v, 0) for v in vals], dtype=np.int64))

    x_cat = torch.tensor(np.array(cat_arrays, dtype=np.int64).T, dtype=torch.long)

    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])
    num_np = df_monitor.select(NUMERICAS_PRECO).fill_null(0.0).fill_nan(0.0).to_numpy().astype(np.float32)
    x_num  = torch.tensor(sc.transform(num_np), dtype=torch.float32)
    y_real = df_monitor["pressao_preco"].fill_null(0.5).fill_nan(0.5).to_numpy()

    # ── reconstrói o modelo ──────────────────────────────────────────────────
    class MLPPreco(nn.Module):
        def __init__(self, vocab_sizes, n_num, embed_dim, hidden, dropout):
            super().__init__()
            self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes])
            dim = len(vocab_sizes) * embed_dim + n_num
            camadas = []
            for h in hidden:
                camadas += [nn.Linear(dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
                dim = h
            camadas += [nn.Linear(dim, 1), nn.Sigmoid()]
            self.rede = nn.Sequential(*camadas)
        def forward(self, x_cat, x_num):
            embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            return self.rede(torch.cat(embeds + [x_num], dim=1))

    modelo = MLPPreco(ckpt["vocab_sizes"], len(NUMERICAS_PRECO),
                      cfg["embed_dim"], cfg["hidden"], cfg["dropout"])
    modelo.load_state_dict(ckpt["model_state"])
    modelo.eval()

    preds = []
    batch = 1024
    for i in range(0, len(x_cat), batch):
        with torch.no_grad():
            preds.append(modelo(x_cat[i:i+batch], x_num[i:i+batch]).numpy().flatten())
    y_pred = np.concatenate(preds)

    mae    = float(np.mean(np.abs(y_real - y_pred)))
    status = "🔴 DRIFT DETECTADO" if mae > LIMIARES["mae_pressao"] else "🟢 OK"
    log.info(
        f"  MAE pressão de preço (meses 11-12): {mae:.4f} | "
        f"limiar: {LIMIARES['mae_pressao']} | {status}"
    )
    return {
        "status":    status,
        "mae":       round(mae, 4),
        "n_amostras": len(df_monitor),
        "limiar":    LIMIARES["mae_pressao"],
    }


def gerar_relatorio(resultados: dict) -> None:
    """Salva um relatório JSON e imprime um resumo no terminal."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saida = DRIFT_DIR / f"drift_report_{ts}.json"

    # determina status geral
    algum_drift = any(
        "DRIFT" in str(r.get("status", ""))
        for r in resultados.values()
    )

    relatorio = {
        "timestamp":     datetime.now().isoformat(),
        "status_geral":  "🔴 RETREINAMENTO RECOMENDADO" if algum_drift else "🟢 MODELOS SAUDÁVEIS",
        "modelos":       resultados,
        "acao":          "Retreine os modelos com dados mais recentes." if algum_drift else "Nenhuma ação necessária.",
    }

    with open(saida, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    # imprime resumo
    print("\n" + "═" * 55)
    print(f"  RELATÓRIO DE DRIFT — {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("═" * 55)
    print(f"  STATUS GERAL: {relatorio['status_geral']}")
    print()
    for nome, res in resultados.items():
        print(f"  {nome.upper()}: {res.get('status', 'N/A')}")
        for k, v in res.items():
            if k not in ("status",) and v is not None:
                print(f"    {k}: {v}")
    print()
    print(f"  Relatório salvo em: {saida}")
    print(f"  AÇÃO: {relatorio['acao']}")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    resultados = {}
    resultados["aeroporto"]    = avaliar_modelo_aeroporto()
    resultados["assentos"]     = avaliar_modelo_assentos()
    resultados["precificacao"] = avaliar_modelo_precificacao()
    resultados["bagagem"]      = avaliar_modelo_bagagem()
    gerar_relatorio(resultados)
