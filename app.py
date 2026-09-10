"""
Interface Streamlit para o pipeline de pre-processamento (apenas filtros lineares)
de imagens de placas veiculares - Pipeline A (global) e Pipeline B (adaptativo),
ambos definidos em preprocessing_pipeline.py.

Rodar com:
    streamlit run app.py
"""

import sqlite3
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import preprocessing_pipeline as pp

IMAGES_DIR = Path("images")
OUT_GLOBAL = Path("images_processed_global")   # saida do Pipeline A
OUT_ADAPTIVE = Path("images_processed")        # saida do Pipeline B (entregavel principal)

st.set_page_config(page_title="Pipeline de Placas - Filtros Lineares", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def list_images():
    return sorted(p.name for p in IMAGES_DIR.glob("*.jpg"))


@st.cache_data
def load_image_bgr(name):
    return cv2.imread(str(IMAGES_DIR / name))


@st.cache_data
def load_mapping_df(pipeline):
    db_path = (OUT_GLOBAL if pipeline == "global" else OUT_ADAPTIVE) / "mapping.db"
    if not db_path.exists():
        return None
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM image_mapping", conn)
    conn.close()
    return df


def rgb_of(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def show_image(container, img, caption):
    """st.image aceita array 2D (cinza) ou 3D (cor); cor precisa ir BGR->RGB antes."""
    to_show = rgb_of(img) if img.ndim == 3 else img
    container.image(to_show, caption=caption, width="stretch")


def crop_with_margin(img, bbox, margin_ratio=0.2):
    """Recorta a regiao do bbox com uma margem proporcional - funciona tanto em
    imagens coloridas (BGR) quanto em cinza, ja que so faz slicing espacial."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin_ratio), int((y2 - y1) * margin_ratio)
    x1, y1, x2, y2 = pp._clamp_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)
    return img[y1:y2, x1:x2]


def metric_delta(label, before, after, fmt="{:.2f}", better="higher"):
    delta = after - before
    good = (delta > 0) if better == "higher" else (delta < 0)
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    color = "#2e8b57" if good or delta == 0 else "#c0392b"
    st.markdown(
        f"**{label}:** {fmt.format(before)} &rarr; {fmt.format(after)} "
        f"<span style='color:{color}'>({arrow} {fmt.format(abs(delta))})</span>",
        unsafe_allow_html=True,
    )


FLAG_LABELS = {
    "baixa_luz": "Baixa luz",
    "estourada": "Superexposta",
    "baixo_contraste": "Baixo contraste",
    "desfocada": "Desfocada",
    "ruidosa": "Ruidosa",
    "placa_pequena": "Placa pequena na cena",
}

PIPELINE_LABELS = {"global": "Pipeline A - global", "adaptive": "Pipeline B - adaptativo"}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline de Pre-processamento")
st.sidebar.caption("Apenas filtros lineares - Pratica 1 de Visao Computacional")
page = st.sidebar.radio("Secao", ["Imagem unica", "Lote (100 imagens)"])

st.sidebar.divider()
st.sidebar.subheader("Modo de processamento")
mode = st.sidebar.radio(
    "Escolha o pipeline",
    ["global", "adaptive", "custom"],
    format_func=lambda m: {"global": "Pipeline A (global, fixo)",
                            "adaptive": "Pipeline B (adaptativo, por diagnostico)",
                            "custom": "Customizado (ajustar manualmente)"}[m],
)

if mode == "custom":
    st.sidebar.caption("Blocos lineares: cinza -> gaussiana -> afim -> unsharp mask")
    smooth_sigma = st.sidebar.slider("Sigma da suavizacao gaussiana (0 = desligada)", 0.0, 5.0, 0.0, step=0.25)
    target_mean = st.sidebar.slider("Media alvo da normalizacao afim", 0.0, 255.0, 127.0, step=1.0)
    target_std = st.sidebar.slider("Desvio padrao alvo da normalizacao afim", 1.0, 100.0, 60.0, step=1.0)
    sharpen_sigma = st.sidebar.slider("Sigma do unsharp mask", 0.0, 5.0, 2.5, step=0.25)
    sharpen_amount = st.sidebar.slider("Intensidade do unsharp mask (0 = desligado)", 0.0, 3.0, 1.0, step=0.1)
    use_plate_stats = st.sidebar.toggle("Ler media/desvio da normalizacao no recorte da placa", value=True)

st.sidebar.divider()
st.sidebar.subheader("Recorte da placa")
show_crop = st.sidebar.toggle("Mostrar recorte da placa", value=True)
crop_margin = st.sidebar.slider("Margem ao redor da placa (%)", 0, 100, 20, step=5) / 100.0


def run_single(img_bgr, bbox):
    """Roda o modo escolhido na sidebar sobre uma imagem e devolve tudo que a
    UI precisa: saida, parametros efetivos, metricas antes/depois, diagnostico
    e a avaliacao pela metrica proposta (ILL)."""
    before = pp.compute_metrics(img_bgr, bbox=bbox)
    flags = pp.diagnose(before)

    if mode == "global":
        out, params, (a, b) = pp.pipeline_global(img_bgr, bbox=bbox)
    elif mode == "adaptive":
        out, params, (a, b) = pp.pipeline_adaptive(img_bgr, bbox=bbox)
    else:
        params_obj = pp.PipelineParams(
            smooth_sigma=smooth_sigma, target_mean=target_mean, target_std=target_std,
            sharpen_sigma=sharpen_sigma, sharpen_amount=sharpen_amount,
        )
        stats_bbox = bbox if (use_plate_stats and bbox is not None) else None
        out, (a, b) = pp.run_linear_pipeline(img_bgr, params_obj, stats_bbox=stats_bbox)
        params = asdict(params_obj)

    after = pp.compute_metrics(out, bbox=bbox)
    evaluation = pp.evaluate_pair(before, after)
    return out, params, (a, b), before, after, flags, evaluation


def show_plate_crop(original, processed, bbox, margin_ratio):
    """Recorte da regiao da placa (antes/depois) - os blocos lineares nao mudam
    a resolucao, entao o bbox e o mesmo antes e depois."""
    st.subheader(f"Recorte da placa (margem de {int(margin_ratio * 100)}%)")
    if bbox is None:
        st.info(
            "Recorte indisponivel: esta imagem nao segue a convencao de nome CCPD "
            "(sem bounding box embutido no nome do arquivo). Funciona para as imagens "
            "do dataset em `images/`."
        )
        return
    c1, c2 = st.columns(2)
    show_image(c1, crop_with_margin(original, bbox, margin_ratio), "original - recorte")
    show_image(c2, crop_with_margin(processed, bbox, margin_ratio), "processada - recorte")


def show_evaluation(evaluation):
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Ganho de contraste", f"{evaluation['contrast_gain']:.2f}x")
    e2.metric("Retencao de nitidez", f"{evaluation['sharpness_retention']:.2f}x")
    e3.metric("Fracao saturada (depois)", f"{evaluation['clipped_fraction_after']:.1%}")
    e4.metric("ILL", f"{evaluation['ill']:.2f}", delta="sucesso" if evaluation["success"] else "nao atingiu o alvo")


# ---------------------------------------------------------------------------
# Pagina: Imagem unica
# ---------------------------------------------------------------------------
if page == "Imagem unica":
    st.title("Pre-processamento de uma imagem")

    images = list_images()
    source = st.radio("Origem da imagem", ["Dataset (images/)", "Enviar arquivo"], horizontal=True)

    if source == "Dataset (images/)":
        filename = st.selectbox("Escolha uma imagem", images)
        img_bgr = load_image_bgr(filename)
        bbox = pp.parse_ccpd_bbox(Path(filename).stem)
    else:
        uploaded = st.file_uploader("Envie uma imagem .jpg/.png", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.info("Envie uma imagem para continuar, ou volte para o dataset.")
            st.stop()
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        bbox = None

    processed, params, (a, b), before_m, after_m, flags, evaluation = run_single(img_bgr, bbox)

    col1, col2 = st.columns(2)
    show_image(col1, img_bgr, "Antes (colorida)")
    show_image(col2, processed, f"Depois ({PIPELINE_LABELS.get(mode, 'customizado')}, cinza)")

    if show_crop:
        st.divider()
        show_plate_crop(img_bgr, processed, bbox, crop_margin)

    st.divider()
    st.subheader("Diagnostico")
    flag_cols = st.columns(len(FLAG_LABELS))
    for col, (flag, label) in zip(flag_cols, FLAG_LABELS.items()):
        col.metric(label, "SIM" if flags.get(flag) else "nao")

    st.subheader("Parametros efetivos (bloco afim resolvido em a, b)")
    st.json({**params, "affine_a": round(a, 4), "affine_b": round(b, 4)})

    st.subheader("Metricas objetivas (antes -> depois, na regiao da placa)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_delta("Brilho medio", before_m["brightness_mean"], after_m["brightness_mean"], better="higher" if flags.get("baixa_luz") else "lower")
    with m2:
        metric_delta("Contraste (desvio padrao)", before_m["contrast_std"], after_m["contrast_std"], better="higher")
    with m3:
        metric_delta("Nitidez relativa", before_m["relative_sharpness"], after_m["relative_sharpness"], better="higher", fmt="{:.3f}")
    with m4:
        metric_delta("Ruido estimado", before_m["noise_estimate"], after_m["noise_estimate"], better="lower", fmt="{:.4f}")

    st.subheader("Metrica proposta - Indice de Legibilidade Linear (ILL)")
    show_evaluation(evaluation)

    with st.expander("Quais filtros lineares foram usados e por que"):
        st.markdown(
            """
- **Cinza**: combinacao linear fixa dos canais BGR (0.114 B + 0.587 G + 0.299 R).
- **Suavizacao** (opcional, `smooth_sigma`): convolucao com kernel Gaussiano.
- **Normalizacao de intensidade**: transformacao afim `T(r) = a*r + b` (reta), com
  `a, b` escolhidos para levar a media/desvio da regiao de referencia (cena inteira no
  Pipeline A, recorte da placa no Pipeline B) para o alvo `target_mean`/`target_std`.
- **Nitidez** (`sharpen_amount`): unsharp mask - combinacao linear com blur Gaussiano.

A cadeia inteira (suavizacao + afim + unsharp) e uma unica composicao de operacoes
lineares - `preprocessing_pipeline.check_linearity()` prova isso numericamente
(testa `T(a*f + b*g) == a*T(f) + b*T(g)`). Nenhuma correcao usa gamma, CLAHE,
non-local means, bilateral, mediana ou IA generativa.
            """
        )

# ---------------------------------------------------------------------------
# Pagina: Lote
# ---------------------------------------------------------------------------
else:
    st.title("Visao geral do lote (100 imagens)")

    lote_pipeline = st.radio(
        "Pipeline", ["adaptive", "global"], format_func=lambda p: PIPELINE_LABELS[p], horizontal=True,
    )
    df = load_mapping_df(lote_pipeline)
    if df is None:
        st.warning(
            f"Nenhum `{(OUT_GLOBAL if lote_pipeline == 'global' else OUT_ADAPTIVE)}/mapping.db` encontrado. "
            "Rode o notebook `vc_pratica1_grupo1.ipynb` (ou `preprocessing_pipeline.process_folder`) primeiro."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imagens processadas", len(df))
    c2.metric("Taxa de sucesso (ILL)", f"{df['success'].mean():.0%}")
    c3.metric("ILL medio", f"{df['ill'].mean():.2f}")
    c4.metric("Fracao saturada media", f"{df['clipped_fraction_after'].mean():.1%}")

    st.subheader("Diagnosticos disparados (multi-rotulo)")
    flag_counts = pd.Series({FLAG_LABELS[f]: int(df[f"flag_{f}"].sum()) for f in FLAG_LABELS})
    st.bar_chart(flag_counts)

    st.subheader("Distribuicao do ILL")
    st.bar_chart(df["ill"].round(1).value_counts().sort_index())

    st.divider()
    st.subheader("Explorar imagens do lote")
    flag_filter = st.multiselect("Filtrar por diagnostico", list(FLAG_LABELS.keys()), format_func=lambda f: FLAG_LABELS[f])
    filtered = df
    for flag in flag_filter:
        filtered = filtered[filtered[f"flag_{flag}"] == 1]

    st.caption(f"{len(filtered)} imagem(ns) apos o filtro")
    choice = st.selectbox("Imagem", filtered["filename"].tolist()) if len(filtered) else None

    if choice:
        rec = filtered[filtered["filename"] == choice].iloc[0]
        original_bgr = load_image_bgr(rec["filename"])
        processed_gray = cv2.imread(rec["processed_path"], cv2.IMREAD_GRAYSCALE)

        col1, col2 = st.columns(2)
        show_image(col1, original_bgr, "original")
        show_image(col2, processed_gray, "processada" + (" - sucesso" if rec["success"] else " - nao atingiu o alvo"))

        if show_crop:
            bbox_lote = pp.parse_ccpd_bbox(Path(choice).stem)
            show_plate_crop(original_bgr, processed_gray, bbox_lote, crop_margin)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_delta("Brilho", rec["brightness_mean_before"], rec["brightness_mean_after"])
        with m2:
            metric_delta("Contraste", rec["contrast_std_before"], rec["contrast_std_after"])
        with m3:
            metric_delta("Nitidez relativa", rec["relative_sharpness_before"], rec["relative_sharpness_after"], fmt="{:.3f}")
        with m4:
            metric_delta("Ruido", rec["noise_estimate_before"], rec["noise_estimate_after"], better="lower", fmt="{:.4f}")

        st.markdown(
            f"**ILL:** {rec['ill']:.2f} &nbsp;|&nbsp; **ganho de contraste:** {rec['contrast_gain']:.2f}x "
            f"&nbsp;|&nbsp; **retencao de nitidez:** {rec['sharpness_retention']:.2f}x"
        )

    st.divider()
    st.subheader("Tabela completa (mapping.db)")
    st.dataframe(df, width="stretch")
