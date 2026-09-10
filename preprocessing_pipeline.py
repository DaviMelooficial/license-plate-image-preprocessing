"""Pipeline de pre-processamento de placas veiculares — SOMENTE filtros lineares.

Blocos disponiveis (todos lineares ou afins, ver ``LINEAR_BLOCKS``):

* conversao para cinza      -> combinacao linear fixa dos canais B, G, R
* suavizacao gaussiana      -> convolucao (linear, invariante ao deslocamento)
* normalizacao afim         -> T(r) = a*r + b, uma reta aplicada pixel a pixel
* unsharp mask              -> I + k*(I - G*I) = convolucao com (1+k)*delta - k*G

Nao ha nenhum operador de ordem (mediana), nenhum peso dependente do conteudo
(bilateral, non-local means), nenhuma equalizacao de histograma (global ou
CLAHE) e nenhuma curva nao-linear de intensidade (gamma, log). Isso e
verificado numericamente em ``check_linearity``.

Dois pipelines usam exatamente os mesmos blocos:

* ``pipeline_global``    -> parametros FIXOS, identicos para as 100 imagens
                             (a pergunta da pratica: um pipeline serve a todas?)
* ``pipeline_adaptive``  -> mesmos blocos, parametros escolhidos pelo
                             diagnostico da imagem e estatisticas do recorte
                             da placa (contraponto para a analise).
"""

import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.restoration import estimate_sigma

CCPD_PATTERN = re.compile(
    r"^(?P<area>\d+)-(?P<tilt_h>\d+)_(?P<tilt_v>\d+)-"
    r"(?P<x1>\d+)&(?P<y1>\d+)_(?P<x2>\d+)&(?P<y2>\d+)-"
    r"(?P<v1x>\d+)&(?P<v1y>\d+)_(?P<v2x>\d+)&(?P<v2y>\d+)_"
    r"(?P<v3x>\d+)&(?P<v3y>\d+)_(?P<v4x>\d+)&(?P<v4y>\d+)-"
    r"(?P<plate_code>[\d_]+)-(?P<brightness_ccpd>\d+)-(?P<blur_ccpd>\d+)$"
)

# Limiares fixos calibrados a partir de mean +/- 0.5*std das metricas do
# RECORTE DA PLACA nas 100 imagens (secao de categorizacao do notebook),
# congelados em valores absolutos para permitir diagnosticar UMA imagem isolada.
BRIGHTNESS_LOW = 50.56
BRIGHTNESS_HIGH = 117.2
CONTRAST_LOW = 18.35
SHARPNESS_LOW = 61.13
NOISE_HIGH = 0.2246
PLATE_AREA_LOW = 0.01825


# --------------------------------------------------------------------------- #
# Metadados e metricas
# --------------------------------------------------------------------------- #
def parse_ccpd_bbox(filename_stem):
    m = CCPD_PATTERN.match(filename_stem)
    if m is None:
        return None
    g = m.groupdict()
    return (int(g["x1"]), int(g["y1"]), int(g["x2"]), int(g["y2"]))


def _clamp_bbox(x1, y1, x2, y2, w, h):
    x1, x2 = sorted((max(0, min(x1, w - 1)), max(0, min(x2, w - 1))))
    y1, y2 = sorted((max(0, min(y1, h - 1)), max(0, min(y2, h - 1))))
    if x2 <= x1:
        x2 = min(x1 + 1, w - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, h - 1)
    return x1, y1, x2, y2


def to_gray(img):
    """BGR -> cinza (0.114 B + 0.587 G + 0.299 R). Ja em cinza: retorna como esta."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def compute_metrics(img, bbox=None):
    """Metricas objetivas na regiao da placa (se bbox) ou na imagem inteira."""
    h, w = img.shape[:2]
    plate_area_ratio = None
    if bbox is not None:
        x1, y1, x2, y2 = _clamp_bbox(*bbox, w, h)
        region = img[y1:y2, x1:x2]
        plate_area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
    else:
        region = img

    gray = to_gray(region)
    noise_sigma = float(estimate_sigma(gray, average_sigmas=True, channel_axis=None))
    clipped = float(np.mean((gray <= 0) | (gray >= 255)))
    contrast = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness_mean": float(gray.mean()),
        "contrast_std": contrast,
        "sharpness_laplacian_var": lap_var,
        # nitidez relativa: desvio do Laplaciano / desvio da imagem. Invariante
        # a T(r) = a*r + b, entao mede ganho de alta frequencia de verdade e
        # nao o simples reescalonamento de intensidade.
        "relative_sharpness": float(np.sqrt(lap_var) / max(contrast, 1e-6)),
        "noise_estimate": noise_sigma,
        "clipped_fraction": clipped,
        "plate_area_ratio": plate_area_ratio,
    }


def diagnose(metrics):
    flags = {
        "baixa_luz": metrics["brightness_mean"] < BRIGHTNESS_LOW,
        "estourada": metrics["brightness_mean"] > BRIGHTNESS_HIGH,
        "baixo_contraste": metrics["contrast_std"] < CONTRAST_LOW,
        "desfocada": metrics["sharpness_laplacian_var"] < SHARPNESS_LOW,
        "ruidosa": metrics["noise_estimate"] > NOISE_HIGH,
        "placa_pequena": False,
    }
    if metrics["plate_area_ratio"] is not None:
        flags["placa_pequena"] = metrics["plate_area_ratio"] < PLATE_AREA_LOW
    return flags


# --------------------------------------------------------------------------- #
# Blocos lineares
# --------------------------------------------------------------------------- #
def gaussian_smooth(gray, sigma):
    """Convolucao com kernel gaussiano. sigma <= 0 desliga o bloco (identidade)."""
    if sigma <= 0:
        return gray.astype(np.float32)
    return cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)


def affine_coefficients(gray, target_mean=127.0, target_std=60.0, stats_region=None):
    """Coeficientes (a, b) da reta T(r) = a*r + b que leva a regiao de
    referencia a ter media/desvio alvo. A regiao e apenas de onde as
    estatisticas sao lidas; a reta e aplicada a imagem inteira."""
    ref = gray if stats_region is None else stats_region
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    a = target_std / max(sd, 1e-6)
    b = target_mean - a * mu
    return a, b


def affine_intensity(gray, a, b):
    """Transformacao linear de intensidade T(r) = a*r + b (sem clipping —
    o clipping para uint8 acontece so na saida, em ``to_uint8``)."""
    return a * gray.astype(np.float32) + b


def unsharp_mask(gray, sigma, amount):
    """I + amount*(I - G_sigma * I). Equivale a UMA convolucao com o kernel
    (1+amount)*delta - amount*G_sigma. amount <= 0 desliga o bloco."""
    g = gray.astype(np.float32)
    if amount <= 0 or sigma <= 0:
        return g
    blurred = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1.0 + amount, blurred, -amount, 0.0)


def to_uint8(img_f32):
    return np.clip(np.rint(img_f32), 0, 255).astype(np.uint8)


LINEAR_BLOCKS = {
    "to_gray": "combinacao linear fixa dos canais (0.114B + 0.587G + 0.299R)",
    "gaussian_smooth": "convolucao com kernel gaussiano normalizado",
    "affine_intensity": "T(r) = a*r + b (reta; a, b sao escalares)",
    "unsharp_mask": "convolucao com (1+k)*delta - k*G_sigma",
}


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
@dataclass
class PipelineParams:
    smooth_sigma: float = 0.0      # 0 = bloco gaussiano desligado
    target_mean: float = 127.0
    target_std: float = 60.0
    sharpen_sigma: float = 2.5
    sharpen_amount: float = 1.0    # 0 = unsharp mask desligado


# Escolhidos por busca em grade sobre as 100 imagens, maximizando a taxa de
# sucesso da metrica proposta (secao "Calibracao" do notebook). A suavizacao
# gaussiana global saiu perdendo em TODAS as configuracoes testadas.
GLOBAL_PARAMS = PipelineParams(smooth_sigma=0.0, target_std=60.0, sharpen_sigma=2.5, sharpen_amount=1.0)


def run_linear_pipeline(img_bgr, params, stats_bbox=None, return_stages=False):
    """Cinza -> suavizacao gaussiana -> normalizacao afim -> unsharp mask.

    ``stats_bbox`` (opcional) restringe de onde a normalizacao afim le media e
    desvio; a transformacao continua sendo aplicada a imagem inteira.
    """
    gray = to_gray(img_bgr)
    smoothed = gaussian_smooth(gray, params.smooth_sigma)

    stats_region = None
    if stats_bbox is not None:
        h, w = gray.shape
        x1, y1, x2, y2 = _clamp_bbox(*stats_bbox, w, h)
        stats_region = smoothed[y1:y2, x1:x2]
    a, b = affine_coefficients(smoothed, params.target_mean, params.target_std, stats_region)
    normalized = affine_intensity(smoothed, a, b)

    sharpened = unsharp_mask(normalized, params.sharpen_sigma, params.sharpen_amount)
    out = to_uint8(sharpened)

    if return_stages:
        stages = {
            "1_cinza": gray,
            "2_gauss": to_uint8(smoothed),
            "3_afim": to_uint8(normalized),
            "4_unsharp": out,
        }
        return out, (a, b), stages
    return out, (a, b)


def pipeline_global(img_bgr, bbox=None):
    """Pipeline A: mesmos parametros para todas as imagens; estatisticas da
    normalizacao lidas da imagem inteira (``bbox`` e ignorado de proposito:
    o pipeline global nao usa a anotacao da placa)."""
    del bbox
    out, ab = run_linear_pipeline(img_bgr, GLOBAL_PARAMS)
    return out, asdict(GLOBAL_PARAMS), ab


ADAPTIVE_BASE = PipelineParams(smooth_sigma=0.0, target_std=45.0, sharpen_sigma=1.5, sharpen_amount=0.0)
ADAPTIVE_SHARPEN_AMOUNT = 1.0


def adaptive_params(flags):
    """Escolha de parametros por diagnostico. So muda ESCALARES dos mesmos
    blocos lineares — nenhum bloco novo entra.

    Base: apenas normalizacao afim (estatisticas da placa, alvo 127 +/- 45).
    Placa desfocada: liga o unsharp mask (sigma 1.5, k = 1.0).
    O ramo "ruidosa -> gaussiana" foi testado e REJEITADO pela metrica
    (ver secao de calibracao do notebook); fica documentado, nao aplicado."""
    p = PipelineParams(**asdict(ADAPTIVE_BASE))
    if flags["desfocada"]:
        p.sharpen_amount = ADAPTIVE_SHARPEN_AMOUNT
    return p


def pipeline_adaptive(img_bgr, bbox=None):
    """Pipeline B: mesmos blocos, parametros por diagnostico e normalizacao
    afim com estatisticas lidas do recorte da placa."""
    metrics = compute_metrics(img_bgr, bbox=bbox)
    flags = diagnose(metrics)
    params = adaptive_params(flags)
    out, ab = run_linear_pipeline(img_bgr, params, stats_bbox=bbox)
    return out, asdict(params), ab


PIPELINES = {"global": pipeline_global, "adaptive": pipeline_adaptive}


# --------------------------------------------------------------------------- #
# Verificacao de linearidade
# --------------------------------------------------------------------------- #
def _spatial_operator(img_f32, params):
    """Parte espacial do pipeline (sem clipping), em float, para o teste de
    linearidade: gauss -> afim(a, b fixos) -> unsharp."""
    s = gaussian_smooth(img_f32, params.smooth_sigma)
    s = affine_intensity(s, 1.0, 0.0)
    return unsharp_mask(s, params.sharpen_sigma, params.sharpen_amount)


def check_linearity(params=GLOBAL_PARAMS, shape=(64, 96), seed=0, tol=1e-3):
    """Testa T(alpha*f + beta*g) == alpha*T(f) + beta*T(g) para a cadeia de
    convolucoes do pipeline. Retorna o erro maximo absoluto."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0, 255, shape).astype(np.float32)
    g = rng.uniform(0, 255, shape).astype(np.float32)
    alpha, beta = 0.7, -1.3
    lhs = _spatial_operator(alpha * f + beta * g, params)
    rhs = alpha * _spatial_operator(f, params) + beta * _spatial_operator(g, params)
    err = float(np.max(np.abs(lhs - rhs)))
    return err, err < tol * 255


def effective_kernel(params=GLOBAL_PARAMS, size=31):
    """Resposta ao impulso da cadeia gauss -> unsharp: como todos os blocos
    espaciais sao convolucoes, a composicao e UM unico kernel."""
    delta = np.zeros((size, size), np.float32)
    delta[size // 2, size // 2] = 1.0
    return _spatial_operator(delta, params)


# --------------------------------------------------------------------------- #
# Metrica de avaliacao
# --------------------------------------------------------------------------- #
SHARPNESS_RETENTION_MIN = 0.9
CLIPPED_MAX = 0.05
BRIGHTNESS_RANGE = (40.0, 215.0)


def kernel_noise_gain(params=GLOBAL_PARAMS):
    """||h||_2 do kernel efetivo: fator pelo qual um filtro linear multiplica o
    desvio-padrao de ruido branco. Como o pipeline e linear, o ganho total de
    ruido e exatamente a * ||h||_2 (a = coeficiente da normalizacao afim)."""
    h = effective_kernel(params)
    return float(np.sqrt(np.sum(h.astype(np.float64) ** 2)))


def evaluate_pair(before, after):
    """Metrica proposta — Indice de Legibilidade Linear (ILL), medida no recorte
    da placa, comparando entrada e saida em dois eixos:

    * eixo 1, contraste:  ganho_contraste = sigma_depois / sigma_antes
    * eixo 2, estrutura:  retencao_nitidez = rho_depois / rho_antes, com
                          rho = sigma(Laplaciano) / sigma(imagem)  (invariante afim)

    e um custo de artefato: fracao de pixels saturados (0 ou 255) e brilho
    medio da placa dentro de uma faixa legivel.

    ILL = log2(ganho_contraste) + log2(min(retencao_nitidez, 1))
    sucesso := ganho_contraste > 1  e  retencao >= 0.9  e  saturacao < 5 %
               e  brilho da placa em [40, 215].
    """
    contrast_gain = after["contrast_std"] / max(before["contrast_std"], 1e-6)
    retention = after["relative_sharpness"] / max(before["relative_sharpness"], 1e-6)
    ill = float(np.log2(max(contrast_gain, 1e-6)) + np.log2(min(max(retention, 1e-6), 1.0)))
    lo, hi = BRIGHTNESS_RANGE
    return {
        "contrast_gain": float(contrast_gain),
        "sharpness_retention": float(retention),
        "clipped_fraction_after": after["clipped_fraction"],
        "ill": ill,
        "success": bool(
            contrast_gain > 1.0
            and retention >= SHARPNESS_RETENTION_MIN
            and after["clipped_fraction"] < CLIPPED_MAX
            and lo <= after["brightness_mean"] <= hi
        ),
    }


# --------------------------------------------------------------------------- #
# Lote
# --------------------------------------------------------------------------- #
def process_folder(images_dir, output_dir, pipeline="global", db_path=None):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fn = PIPELINES[pipeline]

    records = []
    for path in sorted(images_dir.glob("*.jpg")):
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue
        bbox = parse_ccpd_bbox(path.stem)

        before = compute_metrics(img_bgr, bbox=bbox)
        flags = diagnose(before)
        processed, params, (a, b) = fn(img_bgr, bbox=bbox)
        after = compute_metrics(processed, bbox=bbox)
        evaluation = evaluate_pair(before, after)

        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), processed, [cv2.IMWRITE_JPEG_QUALITY, 95])

        records.append({
            "filename": path.name,
            "original_path": str(path),
            "processed_path": str(out_path),
            "pipeline": pipeline,
            **{f"flag_{k}": v for k, v in flags.items()},
            **{f"param_{k}": v for k, v in params.items()},
            "affine_a": a,
            "affine_b": b,
            **{f"{k}_before": v for k, v in before.items()},
            **{f"{k}_after": v for k, v in after.items()},
            **evaluation,
        })

    df = pd.DataFrame(records)
    if db_path is not None:
        conn = sqlite3.connect(db_path)
        df.to_sql("image_mapping", conn, if_exists="replace", index=True, index_label="id")
        conn.close()
    return df
