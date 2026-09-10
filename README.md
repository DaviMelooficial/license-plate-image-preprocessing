# License Plate Image Preprocessing — Prática 1 (Visão Computacional)

Pipeline de pré-processamento de imagens de placas veiculares (dataset no formato CCPD) construído **somente com filtros lineares** — convoluções (gaussiana, unsharp mask) e a transformação de intensidade afim `T(r) = a·r + b` — conforme a política da prática. Nenhuma operação não-linear (mediana, bilateral, non-local means, equalização/CLAHE, gamma, limiarização) é aplicada às imagens; a linearidade da cadeia é verificada numericamente no notebook.

**Equipe:** Ivan Edward, Elizabete Barbosa, Davi Melo

## Entregáveis

| Arquivo | Conteúdo |
|---|---|
| `vc_pratica1_grupo1.ipynb` | Notebook com células executadas: análise inicial, implementação (diagrama de blocos, métrica, calibração) e análise dos resultados. Nomes dos integrantes na primeira célula. |
| `vc_pratica1_grupo1_imagens.zip` | 100 imagens filtradas pelo Pipeline B (adaptativo), mesmos nomes das originais. Gerado pela última seção do notebook. |
| `preprocessing_pipeline.py` | Blocos lineares, os dois pipelines, métrica de avaliação e processamento em lote (importado pelo notebook e pelo `app.py`). |
| `app.py` | Interface Streamlit para explorar os dois pipelines interativamente — uma imagem por vez (com sliders para os parâmetros) ou o lote completo com métricas agregadas. |
| `images/` | Dataset original (100 imagens CCPD, 720×1160). |

## Os dois pipelines

Mesmos quatro blocos lineares (`to_gray → gaussian_smooth → affine_intensity → unsharp_mask`), em `run_linear_pipeline`:

- **Pipeline A — global:** parâmetros fixos para as 100 imagens, sem usar a anotação da placa; estatísticas da afim lidas da cena inteira. Responde à pergunta do enunciado ("um pipeline serve a todas?").
- **Pipeline B — adaptativo:** estatísticas da afim lidas do recorte da placa (bbox do nome do arquivo); unsharp mask só liga quando o diagnóstico indica desfoque. O diagnóstico escolhe apenas escalares — nenhum bloco não-linear entra.

Resultado (métrica proposta, ver abaixo): **Pipeline A ≈ 61% de sucesso, Pipeline B ≈ 88%** — a resposta é "não, um pipeline global não serve igual a todas", e o ganho do B vem quase todo de onde as estatísticas da normalização são lidas (placa vs. cena inteira), não da escolha dos filtros em si. Detalhes e limitações da análise na seção 3.4 do notebook.

## Métrica proposta — Índice de Legibilidade Linear (ILL)

Medida no recorte da placa, entrada vs. saída, em dois eixos: ganho de contraste RMS e retenção de nitidez relativa `ρ = σ(∇²I)/σ(I)` (invariante a transformações afins — evita confundir reescalonamento de intensidade com nitidez real), com custos de artefato (fração saturada < 5 %, brilho da placa em [40, 215]). `ILL = log2(ganho_contraste) + log2(min(retenção, 1))`; uma imagem é "sucesso" se cumpre as quatro condições. Definida antes da calibração; limitações discutidas na seção 3.4 do notebook.

## Diagrama de Blocos da Pipeline

Ver **[DIAGRAMA_BLOCOS.md](DIAGRAMA_BLOCOS.md)** — em arquivo separado porque o Mermaid embutido
direto neste README quebrava a renderização no GitHub. O mesmo diagrama (com os parâmetros
numéricos resolvidos de cada bloco) também aparece como imagem dentro do notebook, seção 2.1.

## Ambiente

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação por política de execução, rode antes (só vale para aquela
janela de terminal): `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Como rodar

Notebook (recria `images_processed/` — Pipeline B —, `images_processed_global/` — Pipeline A — e o `.zip`; ~3 min; as duas pastas são saídas intermediárias e não são versionadas):

```bash
jupyter nbconvert --to notebook --execute --inplace vc_pratica1_grupo1.ipynb
```

Interface interativa (Streamlit) — abre no navegador em `http://localhost:8501`:

```bash
streamlit run app.py
```

A aba "Imagem única" deixa escolher uma imagem do dataset (ou enviar uma nova), rodar o Pipeline A, o B, ou ajustar manualmente os parâmetros dos blocos lineares pela barra lateral, com diagnóstico, antes/depois e métricas (incluindo o ILL) em tempo real. A aba "Lote (100 imagens)" mostra a distribuição de diagnósticos, a taxa de sucesso de cada pipeline e permite navegar pelas 100 imagens já processadas (requer ter rodado o notebook antes, para existirem os `mapping.db` de cada pipeline).
