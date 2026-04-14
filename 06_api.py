"""
=============================================================
  ETAPA 6 — API DE INFERÊNCIA EM TEMPO REAL
  Rodar com: uvicorn 06_api:app --host 0.0.0.0 --port 8000
  Testar em: http://localhost:8000/docs
=============================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from datetime import date
from functools import lru_cache
import hashlib, json as _json
import polars as pl

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

MODELS_DIR   = Path("models")
FEATURES_DIR = Path("data/features")

# ══════════════════════════════════════════════════════════════════════════════
#  MAPEAMENTO DE AEROPORTOS → REGIÕES
# ══════════════════════════════════════════════════════════════════════════════

# Cobre os ~35 maiores aeroportos brasileiros; aceita tanto ICAO (4 letras) quanto IATA (3 letras)
REGIAO_AEROPORTO: dict = {
    # ── Sudeste ───────────────────────────────────────────────────────────────
    "SBGR": "Sudeste", "GRU": "Sudeste",   # São Paulo / Guarulhos
    "SBSP": "Sudeste", "CGH": "Sudeste",   # São Paulo / Congonhas
    "SBKP": "Sudeste", "VCP": "Sudeste",   # Campinas / Viracopos
    "SBRJ": "Sudeste", "SDU": "Sudeste",   # Rio de Janeiro / Santos Dumont
    "SBGL": "Sudeste", "GIG": "Sudeste",   # Rio de Janeiro / Galeão
    "SBBH": "Sudeste", "PLU": "Sudeste",   # Belo Horizonte / Pampulha
    "SBCF": "Sudeste", "CNF": "Sudeste",   # Belo Horizonte / Confins
    "SBVT": "Sudeste", "VIX": "Sudeste",   # Vitória
    "SBUR": "Sudeste", "UDI": "Sudeste",   # Uberlândia
    "SBRP": "Sudeste", "RAO": "Sudeste",   # Ribeirão Preto
    "SBSJ": "Sudeste", "SJK": "Sudeste",   # São José dos Campos
    "SBMK": "Sudeste", "MOC": "Sudeste",   # Montes Claros
    # ── Sul ───────────────────────────────────────────────────────────────────
    "SBPA": "Sul",     "POA": "Sul",       # Porto Alegre
    "SBCT": "Sul",     "CWB": "Sul",       # Curitiba
    "SBFL": "Sul",     "FLN": "Sul",       # Florianópolis
    "SBNF": "Sul",     "NVT": "Sul",       # Navegantes
    "SBLO": "Sul",     "LDB": "Sul",       # Londrina
    "SBMG": "Sul",     "MGF": "Sul",       # Maringá
    "SBFI": "Sul",     "IGU": "Sul",       # Foz do Iguaçu
    "SBCA": "Sul",     "CAC": "Sul",       # Cascavel
    "SBPK": "Sul",     "PEL": "Sul",       # Pelotas
    # ── Nordeste ──────────────────────────────────────────────────────────────
    "SBSV": "Nordeste", "SSA": "Nordeste", # Salvador
    "SBRF": "Nordeste", "REC": "Nordeste", # Recife
    "SBFZ": "Nordeste", "FOR": "Nordeste", # Fortaleza
    "SBNT": "Nordeste", "NAT": "Nordeste", # Natal
    "SBMO": "Nordeste", "MCZ": "Nordeste", # Maceió
    "SBJP": "Nordeste", "JPA": "Nordeste", # João Pessoa
    "SBTE": "Nordeste", "THE": "Nordeste", # Teresina
    "SBSL": "Nordeste", "SLZ": "Nordeste", # São Luís
    "SBIL": "Nordeste", "IOS": "Nordeste", # Ilhéus
    "SBJU": "Nordeste", "JDO": "Nordeste", # Juazeiro do Norte
    "SBPB": "Nordeste", "PHB": "Nordeste", # Parnaíba
    # ── Norte ────────────────────────────────────────────────────────────────
    "SBBE": "Norte",   "BEL": "Norte",    # Belém
    "SBEG": "Norte",   "MAO": "Norte",    # Manaus / Eduardo Gomes
    "SBMQ": "Norte",   "MCP": "Norte",    # Macapá
    "SBRB": "Norte",   "RBR": "Norte",    # Rio Branco
    "SBPV": "Norte",   "PVH": "Norte",    # Porto Velho
    "SBSN": "Norte",   "STM": "Norte",    # Santarém
    "SBBV": "Norte",   "BVB": "Norte",    # Boa Vista
    "SBTF": "Norte",   "TFF": "Norte",    # Tefé
    # ── Centro-Oeste ─────────────────────────────────────────────────────────
    "SBBR": "Centro-Oeste", "BSB": "Centro-Oeste", # Brasília
    "SBCG": "Centro-Oeste", "CGR": "Centro-Oeste", # Campo Grande
    "SBCY": "Centro-Oeste", "CGB": "Centro-Oeste", # Cuiabá
    "SBGO": "Centro-Oeste", "GYN": "Centro-Oeste", # Goiânia
    "SBPP": "Centro-Oeste", "PMG": "Centro-Oeste", # Ponta Porã
}


def obter_regiao(aeroporto: str) -> str:
    """
    Mapeia o código ICAO ou IATA de um aeroporto para a região geográfica brasileira.
    Retorna 'Sudeste' como fallback (maioria das rotas principais) e loga aviso.
    """
    codigo = aeroporto.strip().upper()
    regiao = REGIAO_AEROPORTO.get(codigo)
    if regiao is None:
        log.warning(
            f"Aeroporto '{codigo}' não encontrado no mapeamento de regiões — "
            "usando 'Sudeste' como fallback. Considere atualizar REGIAO_AEROPORTO."
        )
        return "Sudeste"
    return regiao


# Conjunto de todos os códigos ICAO e IATA brasileiros conhecidos.
# flag_int é derivado a partir deste set — mais robusto do que depender do
# campo continente_destino (sujeito a variações de acento e digitação).
AEROPORTOS_BRASIL: set = set(REGIAO_AEROPORTO.keys())


def calcular_flag_int(origem: str, destino: str) -> int:
    """
    Retorna 1 (internacional) se qualquer um dos aeroportos NÃO estiver no
    conjunto de aeroportos brasileiros; 0 (doméstico) se ambos estiverem.
    """
    origem_br  = origem.strip().upper() in AEROPORTOS_BRASIL
    destino_br = destino.strip().upper() in AEROPORTOS_BRASIL
    return 0 if (origem_br and destino_br) else 1


def lookup_stats_rota(
    origem: str,
    destino: str,
    empresa: str,
    parquet_path: Path,
    colunas: list,
    defaults: dict,
) -> dict:
    """
    Busca estatísticas históricas no parquet de features para a rota e empresa informados.

    Estratégia de fallback (da mais para a menos específica):
      1. rota_od + sg_empresa_icao
      2. só rota_od (média entre empresas)
      3. só sg_empresa_icao (média entre rotas)
      4. valores padrão (defaults) com warning logado
    """
    rota = f"{origem}→{destino}"
    try:
        df_lazy = pl.scan_parquet(parquet_path).select(
            ["rota_od", "sg_empresa_icao"] + colunas
        )

        def _media(frame) -> dict | None:
            resultado = frame.select([pl.col(c).mean() for c in colunas]).collect()
            if len(resultado) == 0:
                return None
            row = resultado.row(0, named=True)
            if all(v is None for v in row.values()):
                return None
            return {c: float(row[c]) if row[c] is not None else defaults[c] for c in colunas}

        # tentativa 1: rota + empresa
        stats = _media(
            df_lazy.filter(
                (pl.col("rota_od") == rota) & (pl.col("sg_empresa_icao") == empresa)
            )
        )
        if stats:
            return stats

        # tentativa 2: só rota
        stats = _media(df_lazy.filter(pl.col("rota_od") == rota))
        if stats:
            log.warning(
                f"Stats para rota={rota} empresa={empresa} não encontradas — "
                "usando média da rota entre todas as empresas."
            )
            return stats

        # tentativa 3: só empresa
        stats = _media(df_lazy.filter(pl.col("sg_empresa_icao") == empresa))
        if stats:
            log.warning(
                f"Stats para rota={rota} não encontradas — "
                f"usando média histórica da empresa={empresa}."
            )
            return stats

    except Exception as exc:
        log.warning(f"Erro ao ler {parquet_path} para lookup de stats: {exc}")

    log.warning(
        f"Sem dados históricos para rota={rota} empresa={empresa} em {parquet_path.name} — "
        "usando valores padrão."
    )
    return dict(defaults)


# ── cache de predições em memória ──────────────────────────────────────────
# guarda até 512 resultados — chave = hash dos parâmetros do voo
_cache: dict = {}
_CACHE_MAX = 512

def _cache_key(dados) -> str:
    """Gera uma chave única para os parâmetros do voo."""
    params = f"{dados.aeroporto_origem}|{dados.aeroporto_destino}|{dados.empresa}|{dados.hora_partida}|{dados.mes}|{dados.dia_semana}|{dados.assentos}|{dados.distancia_km}|{dados.continente_destino}"
    return hashlib.md5(params.encode()).hexdigest()

def _get_cache(key: str):
    return _cache.get(key)

def _set_cache(key: str, value: dict):
    if len(_cache) >= _CACHE_MAX:
        # remove o mais antigo (primeiro inserido)
        oldest = next(iter(_cache))
        del _cache[oldest]
    _cache[key] = value



# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITETURAS DOS MODELOS
# ══════════════════════════════════════════════════════════════════════════════

class AtencaoTemporal(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.atencao = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_out):
        scores = self.atencao(lstm_out)
        pesos  = torch.softmax(scores, dim=1)
        return (lstm_out * pesos).sum(dim=1)


class LSTMAeroporto(nn.Module):
    def __init__(self, n_features, horizonte, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.atencao = AtencaoTemporal(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, horizonte * 2),
        )
        self.horizonte = horizonte

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        ctx = self.atencao(lstm_out)
        return self.head(ctx).view(-1, self.horizonte, 2)


class FeatureTokenizer(nn.Module):
    def __init__(self, vocab_sizes, n_numericas, embed_dim):
        super().__init__()
        self.embeddings  = nn.ModuleList([nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes])
        self.num_proj    = nn.Linear(1, embed_dim)
        self.n_numericas = n_numericas

    def forward(self, x_cat, x_num):
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        for i in range(self.n_numericas):
            tokens.append(self.num_proj(x_num[:, i].unsqueeze(1)))
        return torch.stack(tokens, dim=1)


class FTTransformer(nn.Module):
    def __init__(self, vocab_sizes, n_numericas, embed_dim=64, n_heads=8, n_layers=3, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.tokenizer  = FeatureTokenizer(vocab_sizes, n_numericas, embed_dim)
        self.cls_token  = nn.Parameter(torch.randn(1, 1, embed_dim))
        enc_layer       = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cabeca_reg  = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.cabeca_clf  = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 4))

    def forward(self, x_cat, x_num):
        tokens  = self.tokenizer(x_cat, x_num)
        cls     = self.cls_token.expand(tokens.size(0), -1, -1)
        out     = self.transformer(torch.cat([cls, tokens], dim=1))
        cls_out = out[:, 0, :]
        return self.cabeca_reg(cls_out), self.cabeca_clf(cls_out)


class MLPPreco(nn.Module):
    def __init__(self, vocab_sizes, n_num, embed_dim=32, hidden=(256, 128, 64), dropout=0.2):
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


# ══════════════════════════════════════════════════════════════════════════════
#  SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class EntradaVoo(BaseModel):
    aeroporto_origem:   str   = Field(..., example="SBGR")
    aeroporto_destino:  str   = Field(..., example="SBPA")
    empresa:            str   = Field(..., example="TAM")
    data_voo:           date  = Field(..., example="2024-06-14")
    hora_partida:       int   = Field(..., ge=0, le=23)
    mes:                int   = Field(..., ge=1, le=12)
    dia_semana:         str   = Field(..., example="Sexta-feira")
    assentos:           int   = Field(..., gt=0)
    distancia_km:       float = Field(..., gt=0)
    continente_destino: str   = Field("América do Sul")
    semana_ano:         int   = Field(25, ge=1, le=52)


class Predicoes(BaseModel):
    prob_aeroporto_cheio:  float
    lotacao_aeroporto:     str
    taxa_ocupacao_voo:     float
    assentos_vazios_est:   int
    bucket_ocupacao:       str
    pressao_preco:         float
    recomendacao_preco:    str
    prob_excesso_bagagem:  float
    recomendar_cobranca:   bool
    justificativa_bagagem: str
    resumo:                str   # texto explicativo completo da predição


# ══════════════════════════════════════════════════════════════════════════════
#  GERADOR DE RESUMO EXPLICATIVO
# ══════════════════════════════════════════════════════════════════════════════

def gerar_resumo(
    dados: "EntradaVoo",
    prob_aero: float,
    taxa_occ: float,
    vazios: int,
    pressao: float,
    rec_preco: str,
    prob_bag: float,
    cobrar: bool,
) -> str:
    linhas = []

    # ── aeroporto ──────────────────────────────────────────────────────────
    emoji_aero = "🔴" if prob_aero > 0.75 else "🟡" if prob_aero > 0.50 else "🟢"
    detalhe_aero = (
        "Espere filas longas, check-in demorado e portões cheios. Recomende chegar com pelo menos 2h de antecedência."
        if prob_aero > 0.75 else
        "Movimento moderado esperado. Chegue com 1h30 de antecedência para evitar problemas."
        if prob_aero > 0.50 else
        "Aeroporto tranquilo. Chegada padrão de 1h antes do embarque é suficiente."
    )
    linhas.append(
        f"{emoji_aero} AEROPORTO ({int(prob_aero * 100)}% de chance de estar movimentado): {detalhe_aero}"
    )

    # ── assentos ───────────────────────────────────────────────────────────
    emoji_ass = "🔴" if taxa_occ > 0.90 else "🟡" if taxa_occ > 0.70 else "🟢"
    detalhe_ass = (
        f"Voo praticamente lotado — apenas {vazios} assento(s) vazio(s) dos {dados.assentos} ofertados. "
        "Clientes que precisam sentar juntos devem escolher assentos com antecedência."
        if taxa_occ > 0.90 else
        f"Voo bem preenchido com cerca de {vazios} assento(s) vazio(s). "
        "Boa ocupação mas ainda há flexibilidade para seleção de assentos."
        if taxa_occ > 0.70 else
        f"Voo com bastante espaço — estimativa de {vazios} assento(s) vazio(s). "
        "Boa oportunidade para upgrade ou promoções de última hora."
    )
    linhas.append(
        f"{emoji_ass} VOO ({int(taxa_occ * 100)}% de ocupação prevista): {detalhe_ass}"
    )

    # ── precificação ───────────────────────────────────────────────────────
    emoji_preco = {"desconto": "🟢", "normal": "🔵", "premium": "🟡", "máximo": "🔴"}.get(rec_preco, "🔵")
    detalhe_preco = {
        "desconto": (
            f"Índice de pressão baixo ({int(pressao * 100)}/100). A demanda está abaixo do esperado para esta rota e período. "
            "Considere oferecer promoções ou milhas bônus para estimular vendas."
        ),
        "normal": (
            f"Índice de pressão neutro ({int(pressao * 100)}/100). Demanda dentro do padrão histórico. "
            "Mantenha a precificação base sem ajustes significativos."
        ),
        "premium": (
            f"Índice de pressão elevado ({int(pressao * 100)}/100). Alta demanda identificada para esta rota, "
            "dia da semana e horário. Recomendado cobrar acima do preço base — passageiros tendem a aceitar tarifas maiores."
        ),
        "máximo": (
            f"Índice de pressão muito alto ({int(pressao * 100)}/100). Demanda excepcional detectada. "
            "Oportunidade de precificação máxima — período de alta temporada ou rota com baixa oferta."
        ),
    }.get(rec_preco, "")
    linhas.append(f"{emoji_preco} PREÇO (recomendação: {rec_preco.upper()}): {detalhe_preco}")

    # ── bagagem ────────────────────────────────────────────────────────────
    emoji_bag = "🔴" if cobrar else "🟢"
    detalhe_bag = (
        f"Probabilidade de excesso de bagagem: {int(prob_bag * 100)}%. "
        + (
            "Alta incidência histórica nesta rota — recomendado cobrar despacho obrigatório ou comunicar as regras de bagagem claramente no check-in."
            if cobrar else
            "Baixa incidência histórica de excesso. Não é necessário cobrar despacho obrigatório — evite atritar com o passageiro desnecessariamente."
        )
    )
    linhas.append(f"{emoji_bag} BAGAGEM: {detalhe_bag}")

    # ── conclusão geral ────────────────────────────────────────────────────
    n_alertas = sum([prob_aero > 0.6, taxa_occ > 0.85, rec_preco in ("premium", "máximo"), cobrar])
    if n_alertas >= 3:
        linhas.append("⚠️  ATENÇÃO: Voo com múltiplos indicadores de alta demanda. Monitore de perto e prepare equipe para atendimento reforçado.")
    elif n_alertas == 0:
        linhas.append("✅ Voo tranquilo sem alertas significativos. Operação padrão recomendada.")

    return "\n\n".join(linhas)


# ══════════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO DOS MODELOS
# ══════════════════════════════════════════════════════════════════════════════

_modelos: dict = {}
DEMO_MODE: bool = False


@asynccontextmanager
async def lifespan(app):
    global DEMO_MODE
    log.info("Carregando modelos...")
    try:
        ckpt = torch.load(MODELS_DIR / "modelo_aeroporto.pt", map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        m    = LSTMAeroporto(len(ckpt["features"]), cfg["horizonte"], cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
        m.load_state_dict(ckpt["model_state"]); m.eval()
        _modelos["aeroporto"] = {"modelo": m, "ckpt": ckpt}
        log.info("  ✓ Modelo de aeroporto")

        ckpt = torch.load(MODELS_DIR / "modelo_assentos.pt", map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        m    = FTTransformer(ckpt["vocab_sizes"], ckpt["n_numericas"], cfg["embed_dim"], cfg["n_heads"], cfg["n_layers"], cfg["ffn_dim"], cfg["dropout"])
        m.load_state_dict(ckpt["model_state"]); m.eval()
        _modelos["assentos"] = {"modelo": m, "ckpt": ckpt}
        log.info("  ✓ Modelo de assentos")

        ckpt = torch.load(MODELS_DIR / "modelo_precificacao.pt", map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        m    = MLPPreco(ckpt["vocab_sizes"], 7, cfg["embed_dim"], cfg["hidden"], cfg["dropout"])
        m.load_state_dict(ckpt["model_state"]); m.eval()
        _modelos["precificacao"] = {"modelo": m, "ckpt": ckpt}
        log.info("  ✓ Modelo de precificação")

        lgb_bag = lgb.Booster(model_file=str(MODELS_DIR / "lgb_bagagem.txt"))
        ckpt    = torch.load(MODELS_DIR / "dnn_bagagem.pt", map_location="cpu", weights_only=False)
        dnn     = nn.Sequential(
            nn.Linear(ckpt["input_dim"], 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1), nn.Sigmoid(),
        )
        dnn.load_state_dict(ckpt["model_state"]); dnn.eval()
        _modelos["bagagem"] = {"lgb": lgb_bag, "dnn": dnn, "ckpt": ckpt}
        log.info("  ✓ Modelo de bagagem")

        log.info("Todos os modelos carregados!")
    except Exception as exc:
        log.warning(
            f"Arquivos de modelo não encontrados ({exc}). "
            "Iniciando em MODO DEMO — respostas geradas por heurística."
        )
        DEMO_MODE = True
    yield


app = FastAPI(title="ANAC Prediction API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES DE INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

def _encode(valor, classes) -> int:
    """
    Converte um valor categórico para seu índice no vocabulário do encoder.
    Se o valor não for encontrado (OOV), usa o token '__desconhecido__' salvo
    durante o treino em vez de silenciosamente retornar 0 (padding_idx).
    """
    classes_list = list(classes)
    try:
        return classes_list.index(valor)
    except ValueError:
        oov_token = "__desconhecido__"
        if oov_token in classes_list:
            log.warning(
                f"Valor OOV encontrado: '{valor}' — usando token '{oov_token}'."
            )
            return classes_list.index(oov_token)
        log.warning(
            f"Valor OOV encontrado: '{valor}' e token '{oov_token}' não está no "
            "vocabulário — usando índice 0. Considere retreinar com o token OOV."
        )
        return 0


def _resolver_tipo_linha(is_internacional: bool, classes_tipo_linha) -> str:
    """
    Mapeia flag internacional → ds_tipo_linha usando os valores **reais** do
    encoder salvo no checkpoint, evitando mismatch entre treino e inferência.
    Recebe bool para não depender de strings com acento (ex.: "América do Sul").
    """
    is_int = bool(is_internacional)
    classes = list(classes_tipo_linha)

    if is_int:
        for cls in classes:
            if "INTER" in cls.upper():
                return cls
        log.warning(
            "Nenhum valor internacional encontrado no encoder de ds_tipo_linha — "
            "usando 'INTERNACIONAL'."
        )
        return "INTERNACIONAL"
    else:
        # prefere DOMÉSTICA; aceita REGIONAL como alternativa
        for cls in classes:
            if "DOM" in cls.upper():
                return cls
        for cls in classes:
            if "REG" in cls.upper() and "INTER" not in cls.upper():
                return cls
        log.warning(
            "Nenhum valor doméstico encontrado no encoder de ds_tipo_linha — "
            "usando 'DOMÉSTICA'."
        )
        return "DOMÉSTICA"


def inferir_aeroporto(dados: EntradaVoo):
    hora, mes  = dados.hora_partida, dados.mes
    fator_hora = 0.7 if 6 <= hora <= 9 or 17 <= hora <= 20 else 0.4
    fator_mes  = 0.8 if mes in {1, 2, 7, 12} else 0.5
    fator_dia  = 0.75 if dados.dia_semana in {"Sexta-feira", "Domingo"} else 0.5
    prob  = min(0.99, fator_hora * 0.5 + fator_mes * 0.3 + fator_dia * 0.2)
    nivel = "baixa" if prob < 0.35 else "moderada" if prob < 0.60 else "alta" if prob < 0.80 else "crítica"
    return round(prob, 3), nivel


def inferir_assentos(dados: EntradaVoo):
    m, ckpt = _modelos["assentos"]["modelo"], _modelos["assentos"]["ckpt"]
    rota     = f"{dados.aeroporto_origem}→{dados.aeroporto_destino}"
    faixa    = "curta" if dados.distancia_km < 500 else "media" if dados.distancia_km < 1500 else "longa"

    # Usa set AEROPORTOS_BRASIL — não depende de string com acento do campo continente_destino
    flag_int = calcular_flag_int(dados.aeroporto_origem, dados.aeroporto_destino)

    # BUG CRÍTICO 3 — usa os valores reais do encoder em vez de "Internacional"/"Doméstica"
    tipo = _resolver_tipo_linha(
        bool(flag_int),
        ckpt["encoders"].get("ds_tipo_linha", []),
    )

    cats  = [dados.aeroporto_origem, dados.aeroporto_destino, rota, dados.empresa, dados.dia_semana, tipo, faixa]
    x_cat = torch.tensor([[_encode(v, ckpt["encoders"][c]) for v, c in zip(cats, ckpt["categoricas"])]], dtype=torch.long)

    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])

    # BUG CRÍTICO 2 — busca estatísticas históricas reais em vez de usar 0.75 hardcoded
    stats_assentos = lookup_stats_rota(
        origem=dados.aeroporto_origem,
        destino=dados.aeroporto_destino,
        empresa=dados.empresa,
        parquet_path=FEATURES_DIR / "feat_assentos.parquet",
        colunas=["ocupacao_media_historica_rota", "ocupacao_media_empresa_mes"],
        defaults={"ocupacao_media_historica_rota": 0.75, "ocupacao_media_empresa_mes": 0.75},
    )

    num_raw = np.array([[
        dados.mes, dados.semana_ano, dados.hora_partida, dados.assentos,
        dados.distancia_km, flag_int,
        stats_assentos["ocupacao_media_historica_rota"],
        stats_assentos["ocupacao_media_empresa_mes"],
    ]], dtype=np.float32)
    x_num = torch.tensor(sc.transform(num_raw), dtype=torch.float32)

    with torch.no_grad():
        pred_reg, pred_clf = m(x_cat, x_num)
    taxa   = float(pred_reg[0, 0].clamp(0, 1))
    bucket = int(pred_clf.argmax(dim=1)[0])
    vazios = max(0, round(dados.assentos * (1 - taxa)))
    return round(taxa, 3), vazios, ["vazio", "parcial", "quase cheio", "lotado"][bucket]


def inferir_precificacao(dados: EntradaVoo):
    m, ckpt = _modelos["precificacao"]["modelo"], _modelos["precificacao"]["ckpt"]
    rota  = f"{dados.aeroporto_origem}→{dados.aeroporto_destino}"
    cats  = [rota, dados.empresa, dados.dia_semana]
    x_cat = torch.tensor(
        [[_encode(v, ckpt["encoders"][c]) for v, c in zip(cats, ["rota_od", "sg_empresa_icao", "nm_dia_semana_referencia"])]],
        dtype=torch.long,
    )
    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])

    # BUG CRÍTICO 2 — busca estatísticas históricas reais em vez de usar valores fixos
    stats_preco = lookup_stats_rota(
        origem=dados.aeroporto_origem,
        destino=dados.aeroporto_destino,
        empresa=dados.empresa,
        parquet_path=FEATURES_DIR / "feat_precificacao.parquet",
        colunas=["ocupacao_media", "combustivel_medio", "load_factor_medio", "frequencia_voos"],
        defaults={"ocupacao_media": 0.75, "combustivel_medio": 0.0, "load_factor_medio": 0.75, "frequencia_voos": 5.0},
    )

    num_raw = np.array([[
        dados.mes, dados.hora_partida, dados.distancia_km,
        stats_preco["ocupacao_media"],
        stats_preco["combustivel_medio"],
        stats_preco["load_factor_medio"],
        stats_preco["frequencia_voos"],
    ]], dtype=np.float32)
    x_num = torch.tensor(sc.transform(num_raw), dtype=torch.float32)

    with torch.no_grad():
        pressao = float(m(x_cat, x_num)[0, 0])
    rec = "desconto" if pressao < 0.30 else "normal" if pressao < 0.55 else "premium" if pressao < 0.80 else "máximo"
    return round(pressao, 3), rec


def inferir_bagagem(dados: EntradaVoo):
    import pandas as pd
    m, ckpt  = _modelos["bagagem"], _modelos["bagagem"]["ckpt"]
    rota     = f"{dados.aeroporto_origem}→{dados.aeroporto_destino}"
    faixa    = "curta" if dados.distancia_km < 500 else "media" if dados.distancia_km < 1500 else "longa"

    # BUG CRÍTICO 1 — deriva regiões reais a partir dos aeroportos da requisição
    regiao_origem  = obter_regiao(dados.aeroporto_origem)
    regiao_destino = obter_regiao(dados.aeroporto_destino)

    # Usa set AEROPORTOS_BRASIL — não depende de string com acento do campo continente_destino
    flag_int = calcular_flag_int(dados.aeroporto_origem, dados.aeroporto_destino)

    # BUG CRÍTICO 3 — usa os valores reais de ds_tipo_linha (padrão ANAC, maiúsculas)
    tipo_linha = _resolver_tipo_linha(bool(flag_int), classes_tipo_linha=[])

    # BUG CRÍTICO 2 — busca estatísticas históricas reais em vez de 0.30 / 5.0 / 0.25
    stats_bag = lookup_stats_rota(
        origem=dados.aeroporto_origem,
        destino=dados.aeroporto_destino,
        empresa=dados.empresa,
        parquet_path=FEATURES_DIR / "feat_bagagem.parquet",
        colunas=["taxa_excesso_historica_rota", "kg_excesso_medio_rota", "taxa_excesso_historica_empresa"],
        defaults={
            "taxa_excesso_historica_rota": 0.30,
            "kg_excesso_medio_rota": 5.0,
            "taxa_excesso_historica_empresa": 0.25,
        },
    )

    row = pd.DataFrame([{
        "rota_od":                    rota,
        "sg_empresa_icao":            dados.empresa,
        "nm_regiao_origem":           regiao_origem,
        "nm_regiao_destino":          regiao_destino,
        "nm_continente_destino":      dados.continente_destino,
        "flag_internacional":         flag_int,
        "nr_mes_referencia":          dados.mes,
        "nm_dia_semana_referencia":   dados.dia_semana,
        "nr_hora_partida_real":       dados.hora_partida,
        "km_distancia":               dados.distancia_km,
        "faixa_distancia":            faixa,
        "ds_tipo_linha":              tipo_linha,
        "nr_passag_pagos":            int(dados.assentos * 0.8),
        "taxa_excesso_historica_rota":     stats_bag["taxa_excesso_historica_rota"],
        "kg_excesso_medio_rota":           stats_bag["kg_excesso_medio_rota"],
        "taxa_excesso_historica_empresa":  stats_bag["taxa_excesso_historica_empresa"],
    }])
    for col in ["rota_od", "sg_empresa_icao", "nm_regiao_origem", "nm_regiao_destino",
                "nm_continente_destino", "faixa_distancia", "ds_tipo_linha", "nm_dia_semana_referencia"]:
        row[col] = row[col].astype("category")

    prob_lgb = m["lgb"].predict(row)[0]

    sc = StandardScaler()
    sc.mean_  = np.array(ckpt["scaler_mean"])
    sc.scale_ = np.array(ckpt["scaler_scale"])
    num_raw = np.array([[
        flag_int, dados.mes, dados.hora_partida, dados.distancia_km,
        int(dados.assentos * 0.8),
        stats_bag["taxa_excesso_historica_rota"],
        stats_bag["kg_excesso_medio_rota"],
        stats_bag["taxa_excesso_historica_empresa"],
        prob_lgb,
    ]], dtype=np.float32)
    x = torch.tensor(sc.transform(num_raw), dtype=torch.float32)

    with torch.no_grad():
        prob_final = float(m["dnn"](x)[0, 0])
    cobrar = prob_final > 0.55
    if flag_int:
        justif = "Voo internacional: alta probabilidade de excesso"
    elif dados.distancia_km > 1500:
        justif = "Voo longo: passageiros levam mais bagagem"
    elif prob_final > 0.7:
        justif = "Histórico elevado de excesso nesta rota"
    elif prob_final < 0.3:
        justif = "Baixo histórico de excesso nesta rota"
    else:
        justif = "Probabilidade moderada — avaliar pelo histórico da empresa"
    return round(prob_final, 3), cobrar, justif


# ══════════════════════════════════════════════════════════════════════════════
#  MODO DEMO — heurística quando os modelos não estão disponíveis
# ══════════════════════════════════════════════════════════════════════════════

def _prever_demo(dados: EntradaVoo) -> dict:
    """Retorna predições realistas geradas por heurística quando os modelos não estão disponíveis."""
    flag_int = calcular_flag_int(dados.aeroporto_origem, dados.aeroporto_destino)
    hora     = dados.hora_partida
    mes      = dados.mes
    dia      = dados.dia_semana
    pico     = 6 <= hora <= 9 or 17 <= hora <= 20
    alta_temporada = mes in {1, 2, 7, 12}
    fim_semana     = dia in {"Sexta-feira", "Domingo"}

    # aeroporto — mesma lógica de inferir_aeroporto (já é heurística)
    prob_aero, nivel_aero = inferir_aeroporto(dados)

    # assentos
    base_occ = 0.82 if flag_int else 0.74
    taxa_occ = round(min(0.99, base_occ + (0.08 if pico else 0) + (0.07 if alta_temporada else 0) + (0.05 if fim_semana else 0)), 3)
    vazios   = max(0, round(dados.assentos * (1 - taxa_occ)))
    bucket   = "lotado" if taxa_occ > 0.90 else "quase cheio" if taxa_occ > 0.70 else "parcial" if taxa_occ > 0.40 else "vazio"

    # precificação
    base_preco = 0.65 if flag_int else 0.45
    pressao    = round(min(0.99, base_preco + (0.10 if alta_temporada else 0) + (0.08 if fim_semana else 0) + (0.06 if pico else 0)), 3)
    rec_preco  = "desconto" if pressao < 0.30 else "normal" if pressao < 0.55 else "premium" if pressao < 0.80 else "máximo"

    # bagagem
    base_bag = 0.35 if flag_int else 0.12
    bag_dist = 0.10 if dados.distancia_km > 1500 else 0.05 if dados.distancia_km > 500 else 0.0
    prob_bag = round(min(0.95, base_bag + bag_dist), 3)
    cobrar   = prob_bag > 0.55
    if flag_int:
        justif = "Voo internacional: alta probabilidade de excesso"
    elif dados.distancia_km > 1500:
        justif = "Voo longo: passageiros levam mais bagagem"
    elif prob_bag < 0.3:
        justif = "Baixo histórico de excesso nesta rota"
    else:
        justif = "Probabilidade moderada — avaliar pelo histórico da empresa"

    resumo = gerar_resumo(dados, prob_aero, taxa_occ, vazios, pressao, rec_preco, prob_bag, cobrar)
    return dict(
        prob_aeroporto_cheio  = prob_aero,
        lotacao_aeroporto     = nivel_aero,
        taxa_ocupacao_voo     = taxa_occ,
        assentos_vazios_est   = vazios,
        bucket_ocupacao       = bucket,
        pressao_preco         = pressao,
        recomendacao_preco    = rec_preco,
        prob_excesso_bagagem  = prob_bag,
        recomendar_cobranca   = cobrar,
        justificativa_bagagem = justif,
        resumo                = resumo,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/prever", response_model=Predicoes)
async def prever(dados: EntradaVoo):
    # modo demo: retorna heurística sem carregar modelos
    if DEMO_MODE:
        resultado = _prever_demo(dados)
        return Predicoes(**resultado)

    # verifica cache antes de rodar os modelos
    key = _cache_key(dados)
    cached = _get_cache(key)
    if cached:
        log.info(f"Cache hit: {dados.aeroporto_origem}→{dados.aeroporto_destino}")
        return Predicoes(**cached)

    try:
        prob_aero,   nivel_aero  = inferir_aeroporto(dados)
        taxa_occ, vazios, bucket = inferir_assentos(dados)
        pressao,     rec_preco   = inferir_precificacao(dados)
        prob_bag, cobrar, justif = inferir_bagagem(dados)
    except Exception as e:
        log.error(f"Erro na inferência: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    resumo = gerar_resumo(dados, prob_aero, taxa_occ, vazios, pressao, rec_preco, prob_bag, cobrar)

    resultado = dict(
        prob_aeroporto_cheio  = prob_aero,
        lotacao_aeroporto     = nivel_aero,
        taxa_ocupacao_voo     = taxa_occ,
        assentos_vazios_est   = vazios,
        bucket_ocupacao       = bucket,
        pressao_preco         = pressao,
        recomendacao_preco    = rec_preco,
        prob_excesso_bagagem  = prob_bag,
        recomendar_cobranca   = cobrar,
        justificativa_bagagem = justif,
        resumo                = resumo,
    )
    _set_cache(key, resultado)
    return Predicoes(**resultado)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "modelos": list(_modelos.keys()),
        "todos_prontos": len(_modelos) == 4,
        "demo_mode": DEMO_MODE,
    }




@app.get("/cache/stats", summary="Estatísticas do cache")
async def cache_stats():
    return {
        "entradas": len(_cache),
        "capacidade_maxima": _CACHE_MAX,
        "uso_pct": round(len(_cache) / _CACHE_MAX * 100, 1),
    }

@app.delete("/cache", summary="Limpa o cache")
async def limpar_cache():
    _cache.clear()
    return {"status": "cache limpo"}

@app.get("/historico", summary="Histórico de ocupação por rota")
async def historico(origem: str, destino: str):
    """Retorna a ocupação média por mês para uma rota. Usado pelo dashboard."""
    try:
        rota = f"{origem.upper()}→{destino.upper()}"
        df   = pl.scan_parquet(FEATURES_DIR / "feat_assentos.parquet")
        dados = (
            df.filter(pl.col("rota_od") == rota)
            .group_by("nr_mes_referencia")
            .agg([
                pl.col("taxa_ocupacao").mean().alias("ocupacao_media"),
                pl.col("nr_passag_pagos").sum().alias("total_passageiros"),
                pl.len().alias("total_voos"),
            ])
            .sort("nr_mes_referencia")
            .collect()
        )
        if len(dados) == 0:
            dados = (
                pl.scan_parquet(FEATURES_DIR / "feat_assentos.parquet")
                .group_by("nr_mes_referencia")
                .agg(pl.col("taxa_ocupacao").mean().alias("ocupacao_media"))
                .sort("nr_mes_referencia")
                .collect()
            )
            encontrada = False
        else:
            encontrada = True
        meses_dict = {int(r["nr_mes_referencia"]): float(r["ocupacao_media"]) for r in dados.iter_rows(named=True)}
        meses = [{"mes": m, "ocupacao_media": round(meses_dict.get(m, 0.0), 3)} for m in range(1, 13)]
        return {"rota": rota, "encontrada": encontrada, "meses": meses, "nota": "Médias sobre todos os anos do dataset."}
    except Exception as e:
        log.error(f"Erro no histórico: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)