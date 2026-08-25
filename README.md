# License Plate Image Preprocessing

Pipeline de pré-processamento de imagens de placas veiculares (dataset no formato CCPD), com o objetivo de melhorar a qualidade de cada imagem de acordo com o problema específico que ela apresenta (baixa luz, superexposição, baixo contraste, desfoque, ruído ou placa pequena na cena), em vez de aplicar uma transformação genérica igual para todas.

## Estrutura

- `eda_pre_process.ipynb` — EDA sobre as 100 imagens: parsing dos metadados embutidos no nome dos arquivos (padrão CCPD), cálculo de métricas objetivas de qualidade (brilho, contraste, nitidez via variância do Laplaciano, ruído via estimador wavelet), categorização das imagens por problema dominante e comparação quantitativa de técnicas candidatas de correção.
- `preprocessing_pipeline.py` — módulo modular com as funções de diagnóstico (`compute_metrics`, `diagnose`) e de correção (`auto_gamma_correction`, `apply_clahe`, `denoise_nlmeans`, `unsharp_mask`, `upscale_lanczos`), mais os orquestradores `process_image` (uma imagem) e `process_folder` (lote completo).
- `pre_process.ipynb` — roda o pipeline sobre `images/`, salva os resultados em `images_processed/` e valida a saída (contagem de arquivos, esquema do banco, distribuição das correções aplicadas).
- `images/` — dataset original (100 imagens).
- `images_processed/` — imagens processadas + `mapping.db` (SQLite) com a relação imagem original ↔ processada, diagnóstico e métricas antes/depois.
- `Pratica1_VC.pdf` — enunciado da prática.

## Ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Metodologia

Os limiares de diagnóstico (`BRIGHTNESS_LOW`, `BRIGHTNESS_HIGH`, `CONTRAST_LOW`, `SHARPNESS_LOW`, `NOISE_HIGH`, `PLATE_AREA_LOW`) foram calibrados a partir da distribuição real das 100 imagens (média ± 0,5 desvio-padrão de cada métrica), documentados em `eda_pre_process.ipynb` e congelados como constantes em `preprocessing_pipeline.py` para permitir avaliar uma imagem isolada sem depender do lote inteiro.

Cada imagem pode disparar mais de uma correção — elas são aplicadas em cadeia, na ordem: denoise → correção de exposição (gamma) → contraste (CLAHE) → nitidez (unsharp mask) → upscale. Todas as correções são validadas com métricas objetivas de antes/depois, não apenas inspeção visual.
