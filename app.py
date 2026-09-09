"""
Interface Streamlit para o pipeline de pre-processamento (apenas filtros lineares)
de imagens de placas veiculares.

Rodar com:
    streamlit run app.py
"""

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import preprocessing_pipeline as pp

IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("images_processed")
DB_PATH = OUTPUT_DIR / "mapping.db"

st.set_page_config(page_title="Pipeline de Placas - Filtros Lineares", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data
def list_images():
    return sorted(p.name for p in IMAGES_DIR.glob("*.jpg"))


@st.cache_data
def load_image(name):
    img = cv2.imread(str(IMAGES_DIR / name))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data
def load_mapping_df():
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM image_mapping", conn)
    conn.close()
    return df


def bgr_of(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)


def rgb_of(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Pipeline de Pre-processamento")
st.sidebar.caption("Apenas filtros lineares - Pratica 1 de Visao Computacional")
page = st.sidebar.radio("Secao", ["Imagem unica", "Lote (100 imagens)"])

st.sidebar.divider()
st.sidebar.subheader("Parametros da pipeline")
auto_mode = st.sidebar.toggle("Modo automatico (diagnostico decide)", value=True)

target_mean = st.sidebar.slider("Brilho alvo", 60, 180, 115, step=5)
low_pct, high_pct = st.sidebar.slider("Percentis do alargamento de contraste", 0, 100, (2, 98))
max_gain = st.sidebar.slider("Ganho maximo do contraste", 1.0, 10.0, 4.0, step=0.5)
denoise_ksize = st.sidebar.select_slider("Kernel do denoise Gaussiano", options=[3, 5, 7, 9], value=5)
unsharp_sigma = st.sidebar.slider("Sigma do unsharp mask", 0.5, 6.0, 3.0, step=0.5)
unsharp_amount = st.sidebar.slider("Intensidade do unsharp mask", 0.0, 3.0, 1.5, step=0.1)
upscale_factor = st.sidebar.slider("Fator de upscale (Lanczos)", 1.0, 3.0, 1.5, step=0.1)

if not auto_mode:
    st.sidebar.caption("Selecione manualmente as correcoes a aplicar:")
    manual_flags = {
        flag: st.sidebar.checkbox(label, value=False)
        for flag, label in FLAG_LABELS.items()
    }
else:
    manual_flags = None


def run_pipeline(img_bgr, bbox=None):
    """Roda o diagnostico + as correcoes lineares com os parametros da sidebar."""
    before_metrics = pp.compute_metrics(img_bgr, bbox=bbox)
    flags = pp.diagnose(before_metrics) if auto_mode else dict(manual_flags)
    if not auto_mode:
        flags.setdefault("placa_pequena", False)

    processed = img_bgr
    corrections = []

    if flags.get("ruidosa"):
        processed = pp.denoise_gaussian(processed, ksize=denoise_ksize)
        corrections.append("ruidosa")
    if flags.get("baixa_luz") or flags.get("estourada"):
        processed = pp.linear_brightness_correction(processed, target_mean=target_mean)
        corrections.append("baixa_luz" if flags.get("baixa_luz") else "estourada")
    if flags.get("baixo_contraste"):
        processed = pp.linear_contrast_stretch(
            processed, low_pct=low_pct, high_pct=high_pct, max_gain=max_gain
        )
        corrections.append("baixo_contraste")
    if flags.get("desfocada"):
        processed = pp.unsharp_mask(processed, sigma=unsharp_sigma, amount=unsharp_amount)
        corrections.append("desfocada")
    if flags.get("placa_pequena"):
        processed = pp.upscale_lanczos(processed, scale=upscale_factor)
        corrections.append("placa_pequena")

    after_bbox = bbox
    if bbox is not None and "placa_pequena" in corrections:
        x1, y1, x2, y2 = bbox
        after_bbox = (
            int(x1 * upscale_factor), int(y1 * upscale_factor),
            int(x2 * upscale_factor), int(y2 * upscale_factor),
        )
    after_metrics = pp.compute_metrics(processed, bbox=after_bbox)

    return processed, corrections, flags, before_metrics, after_metrics


# ---------------------------------------------------------------------------
# Pagina: Imagem unica
# ---------------------------------------------------------------------------
if page == "Imagem unica":
    st.title("Pre-processamento de uma imagem")

    images = list_images()
    col_src1, col_src2 = st.columns([2, 1])
    with col_src1:
        source = st.radio("Origem da imagem", ["Dataset (images/)", "Enviar arquivo"], horizontal=True)

    if source == "Dataset (images/)":
        filename = st.selectbox("Escolha uma imagem", images)
        rgb_img = load_image(filename)
        bbox = pp.parse_ccpd_bbox(Path(filename).stem)
    else:
        uploaded = st.file_uploader("Envie uma imagem .jpg/.png", type=["jpg", "jpeg", "png"])
        if uploaded is None:
            st.info("Envie uma imagem para continuar, ou volte para o dataset.")
            st.stop()
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        bgr_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        rgb_img = rgb_of(bgr_img)
        bbox = None

    img_bgr = bgr_of(rgb_img)
    processed_bgr, corrections, flags, before_m, after_m = run_pipeline(img_bgr, bbox=bbox)
    processed_rgb = rgb_of(processed_bgr)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Antes")
        st.image(rgb_img, width="stretch")
    with col2:
        title = "Depois: " + (", ".join(FLAG_LABELS.get(c, c) for c in corrections) if corrections else "sem correcao (boa qualidade)")
        st.subheader(title)
        st.image(processed_rgb, width="stretch")

    st.divider()
    st.subheader("Diagnostico")
    flag_cols = st.columns(len(FLAG_LABELS))
    for col, (flag, label) in zip(flag_cols, FLAG_LABELS.items()):
        active = bool(flags.get(flag))
        col.metric(label, "SIM" if active else "nao", delta=None)

    st.subheader("Metricas objetivas (antes -> depois)")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_delta("Brilho medio", before_m["brightness_mean"], after_m["brightness_mean"], better="higher" if flags.get("baixa_luz") else "lower")
    with m2:
        metric_delta("Contraste (desvio padrao)", before_m["contrast_std"], after_m["contrast_std"], better="higher")
    with m3:
        metric_delta("Nitidez (var. Laplaciano)", before_m["sharpness_laplacian_var"], after_m["sharpness_laplacian_var"], better="higher", fmt="{:.1f}")
    with m4:
        metric_delta("Ruido estimado", before_m["noise_estimate"], after_m["noise_estimate"], better="lower", fmt="{:.4f}")

    with st.expander("Quais filtros lineares foram usados e por que"):
        st.markdown(
            """
- **Denoise** (`ruidosa`): convolucao com kernel Gaussiano - filtro linear.
- **Brilho** (`baixa_luz` / `estourada`): deslocamento aditivo `g = f + beta` no canal de
  luminancia (Y, espaco YCrCb) - transformacao afim.
- **Contraste** (`baixo_contraste`): alargamento afim por percentis `g = (f - lo) * ganho`,
  tambem so no canal Y, com ganho limitado para nao amplificar ruido de quantizacao.
- **Nitidez** (`desfocada`): unsharp mask - combinacao linear de imagem + blur Gaussiano.
- **Upscale** (`placa_pequena`): interpolacao de Lanczos - filtro linear.

Nenhuma correcao usa gamma, CLAHE, non-local means, bilateral, mediana ou IA generativa.
            """
        )

# ---------------------------------------------------------------------------
# Pagina: Lote
# ---------------------------------------------------------------------------
else:
    st.title("Visao geral do lote (100 imagens)")

    df = load_mapping_df()
    if df is None:
        st.warning(
            "Nenhum `images_processed/mapping.db` encontrado. Rode o notebook "
            "`vc_pratica1_grupo1.ipynb` (ou `preprocessing_pipeline.process_folder`) primeiro."
        )
        st.stop()

    n_corrections = df["corrections_applied"].apply(lambda s: 0 if s == "" else len(s.split(",")))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Imagens processadas", len(df))
    c2.metric("Sem nenhuma correcao", int((n_corrections == 0).sum()))
    c3.metric("Media de correcoes/imagem", f"{n_corrections.mean():.2f}")
    c4.metric("Maximo de correcoes numa imagem", int(n_corrections.max()))

    st.subheader("Distribuicao de correcoes aplicadas")
    st.bar_chart(n_corrections.value_counts().sort_index())

    st.subheader("Taxa de sucesso da metrica proposta, por tipo de correcao")
    checks = {
        "baixa_luz": ("brightness_mean", lambda b, a: a > b),
        "estourada": ("brightness_mean", lambda b, a: a < b),
        "baixo_contraste": ("contrast_std", lambda b, a: a > b),
        "desfocada": ("sharpness_laplacian_var", lambda b, a: a > b),
        "ruidosa": ("noise_estimate", lambda b, a: a < b),
    }
    rows = []
    for flag, (metric, better) in checks.items():
        applied = df[df[f"flag_{flag}"] == 1]
        if applied.empty:
            continue
        success = better(applied[f"{metric}_before"], applied[f"{metric}_after"])
        rows.append({
            "correcao": FLAG_LABELS[flag],
            "n_imagens": len(applied),
            "taxa_sucesso": success.mean(),
        })
    eval_df = pd.DataFrame(rows).set_index("correcao")
    st.bar_chart(eval_df["taxa_sucesso"])
    st.dataframe(eval_df.style.format({"taxa_sucesso": "{:.0%}"}), width="stretch")

    st.divider()
    st.subheader("Explorar imagens do lote")
    flag_filter = st.multiselect("Filtrar por diagnostico", list(FLAG_LABELS.keys()), format_func=lambda f: FLAG_LABELS[f])
    filtered = df
    for flag in flag_filter:
        filtered = filtered[filtered[f"flag_{flag}"] == 1]

    st.caption(f"{len(filtered)} imagem(ns) apos o filtro")
    choice = st.selectbox("Imagem", filtered["original_filename"].tolist()) if len(filtered) else None

    if choice:
        rec = filtered[filtered["original_filename"] == choice].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            st.image(rgb_of(cv2.imread(rec["original_path"])), caption="original", width="stretch")
        with col2:
            st.image(rgb_of(cv2.imread(rec["processed_path"])), caption=f"processada ({rec['corrections_applied'] or 'sem correcao'})", width="stretch")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_delta("Brilho", rec["brightness_mean_before"], rec["brightness_mean_after"])
        with m2:
            metric_delta("Contraste", rec["contrast_std_before"], rec["contrast_std_after"])
        with m3:
            metric_delta("Nitidez", rec["sharpness_laplacian_var_before"], rec["sharpness_laplacian_var_after"], fmt="{:.1f}")
        with m4:
            metric_delta("Ruido", rec["noise_estimate_before"], rec["noise_estimate_after"], better="lower", fmt="{:.4f}")

    st.divider()
    st.subheader("Tabela completa (mapping.db)")
    st.dataframe(df, width="stretch")
