## Project Python referência: https://realpython.com/face-recognition-with-python/
## GitRoot Repositório: https://github.com/realpython/materials/tree/master/face-recognition 
## GitRepo Code: https://github.com/realpython/materials/tree/master/face-recognition/source_code_final

## GQM Métricas — Implementação prática para o projeto de Reconhecimento Facial 

Este é um pacote que provê 
This package provides a **practical implementation** of the Goal–Question–Metric (GQM) plan discussed in chat.

## What it measures

**G1 — Corretude**
- Accuracy, precision, recall, F1
- Confusion matrix
- Overall FPR considering "unknown" as the negative class

**G2 — Desempenho & Recursos**
- Latência por imagem (média, p50, p95, max, min)
- Throughput (img/s)
- Tempo total de execução
- CPU média/máx e RSS (MB) via `psutil`

**G3 — Qualidade de Código**
- Complexidade ciclomática por `radon cc -j`
- Cobertura de testes com `pytest --cov` (se houver `tests/`)

**G4 — Robustez (opcional)**
- Você pode introduzir condições nos nomes/paths dos arquivos (ex.: `_lowlight`, `_angle45`) e depois filtrar a partir do CSV gerado.

## Dataset esperado

```
known/
  alice/*.jpg
  bob/*.jpg
test/
  alice/*.jpg
  bob/*.jpg
  unknown/*.jpg   # opcional (negativos)
```

As pastas em `known/` definem as **classes (labels)**. Em `test/`, se existir a pasta `unknown/`, ela será tratada como negativos.

## Uso

1) Crie um virtualenv e instale dependências:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Execute o coletor apontando para:
- `--repo-path`: o diretório do código a ser inspecionado (por ex. `source_code_final` do RealPython);
- `--known-dir`: base de rostos rotulados;
- `--test-dir`: base de testes;

```bash
python gqm_runner.py \
  --repo-path /path/to/source_code_final \
  --known-dir /path/to/known \
  --test-dir  /path/to/test \
  --out-dir   ./gqm_out \
  --model hog \
  --tolerance 0.6
```

3) Saídas
- `gqm_out/gqm_report.json` — relatório completo (métricas G1, G2, G3, G4)
- `gqm_out/predictions.csv` — pares (y_true, y_pred) por imagem

## Parâmetros para ser preciso nos testes
- Para **CNN mais precisa**, use `--model cnn` (exige GPU/CUDA para render bom desempenho).
- Ajuste `--tolerance` (0.4–0.6 costuma ser razoável; menor = mais estrito).
- Se o repositório não tiver testes, a etapa de cobertura será apenas informativa.
- Para datasets grandes, considere aumentar `--resource-sample-interval`.

### Quebra automática por condições
O `gqm_runner.py` detecta condições nos nomes dos arquivos de teste utilizando sufixos como:
`_lowlight`, `_backlight`, `_blur`, `_occluded`, `_noisy`, `_angle15`, `_angle30`, `_angle45`, `_lowres`, `_highres`.

Exemplo de nome: `alice_01_lowlight.jpg` ou `bob_03_angle45.png`.
