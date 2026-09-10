# License Plate Image Preprocessing — Prática 1 (Visão Computacional)

Pipeline de pré-processamento de imagens de placas veiculares (dataset no formato CCPD) construído **somente com filtros lineares** — convoluções (gaussiana, unsharp mask) e a transformação de intensidade afim `T(r) = a·r + b` — conforme a política da prática. Nenhuma operação não-linear (mediana, bilateral, non-local means, equalização/CLAHE, gamma, limiarização) é aplicada às imagens; a linearidade da cadeia é verificada numericamente no notebook.

## Entregáveis

| Arquivo | Conteúdo |
|---|---|
| `vc_pratica1_davi_melo.ipynb` | Notebook com células executadas: análise inicial, implementação (diagrama de blocos, métrica, calibração) e análise dos resultados. Nomes dos integrantes na primeira célula. |
| `vc_pratica1_davi_melo_imagens.zip` | 100 imagens filtradas pelo Pipeline B, mesmos nomes das originais. Gerado pela última seção do notebook. |
| `preprocessing_pipeline.py` | Blocos lineares, os dois pipelines, métrica de avaliação e processamento em lote (importado pelo notebook). |
| `images/` | Dataset original (100 imagens CCPD, 720×1160). |

## Os dois pipelines

Mesmos quatro blocos lineares (`to_gray → gaussian_smooth → affine_intensity → unsharp_mask`), em `run_linear_pipeline`:

- **Pipeline A — global:** parâmetros fixos para as 100 imagens, sem usar a anotação da placa; estatísticas da afim lidas da cena inteira. Responde à pergunta do enunciado ("um pipeline serve a todas?").
- **Pipeline B — adaptativo:** estatísticas da afim lidas do recorte da placa (bbox do nome do arquivo); unsharp mask só liga quando o diagnóstico indica desfoque. O diagnóstico escolhe apenas escalares — nenhum bloco não-linear entra.

## Métrica proposta — Índice de Legibilidade Linear (ILL)

Medida no recorte da placa, entrada vs. saída, em dois eixos: ganho de contraste RMS e retenção de nitidez relativa `ρ = σ(∇²I)/σ(I)` (invariante a transformações afins), com custos de artefato (fração saturada < 5 %, brilho da placa em [40, 215]). `ILL = log2(ganho_contraste) + log2(min(retenção, 1))`; uma imagem é "sucesso" se cumpre as quatro condições. Definida antes da calibração; limitações discutidas na seção 2.2 do notebook.

## Reproduzir

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab vc_pratica1_davi_melo.ipynb   # Run All
```

A execução completa (~3 min) recria `images_processed/` (Pipeline B), `images_processed_global/` (Pipeline A) e o `.zip`. As duas pastas são saídas intermediárias e não são versionadas.
