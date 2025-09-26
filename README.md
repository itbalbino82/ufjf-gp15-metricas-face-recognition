# Estrutura do Repositório

Abaixo está uma explicação dos principais diretórios e arquivos deste repositório:

| Pasta/Arquivo                | Descrição                                                                                 |
|------------------------------|------------------------------------------------------------------------------------------|
| `código_fonte_opensource/`   | Código original da aplicação. Todo baseado nos links de referência abaixo.               |
| `script_para_testar/`        | Script em Python como exemplo para testes de performance, eficácia e etc.                |
| `script_para_testar/gqm_runner.py`| Script principal para coleta e análise das métricas GQM.                            |
| `script_para_testar/requirements.txt`| Lista de dependências Python necessárias para rodar o projeto.                   |
| `gqm_out/`                   | Pasta gerada com os relatórios e resultados das métricas após execução do Script         |
| `README.md`                  | Este arquivo de documentação.                                                            |
| `Relatório_GQM_Aplicado.md`  | Exemplo de resultados esperados após aplicação do GQM no software.                       |
| `Trabalho_Metric_Software_v1.md` | Documento detalhando o contexto, metas e métricas do trabalho em formato markdown.   |
| `Trabalho_Metric_Software_v1.docx` | Documento detalhando o contexto, metas e métricas do trabalho em formato markdown. |

---

# Reconhecimento Facial — Métricas GQM

> **Referências:**
> - [Tutorial RealPython](https://realpython.com/face-recognition-with-python/)
> - [Repositório GitRoot](https://github.com/realpython/materials/tree/master/face-recognition)
> - [Código Final](https://github.com/realpython/materials/tree/master/face-recognition/source_code_final)

---

## 📊 GQM Métricas — Implementação Prática

Este pacote fornece uma **implementação prática** do plano Goal–Question–Metric (GQM) para projetos de reconhecimento facial.

---

## O que é Medido?

| Meta         | Métricas Principais                                                                 |
|--------------|-------------------------------------------------------------------------------------|
| **G1 — Eficacia**      | Acurácia, Precisão, Recall, F1, Confusion Matrix, FPR (classe "unknown")         |
| **G2 — Desempenho**     | Latência (média, p50, p95, max, min), Throughput (img/s), Tempo total, CPU/RSS    |
| **G3 — Qualidade Código**| Complexidade ciclomática (`radon cc -j`), Cobertura de testes (`pytest --cov`)   |
| **G4 — Robustez**       | Filtragem por condições nos nomes dos arquivos (ex.: `_lowlight`, `_angle45`)     |

---

## 📁 Estrutura Esperada do Dataset

```
known/
  alice/*.jpg
  bob/*.jpg
test/
  alice/*.jpg
  bob/*.jpg
  unknown/*.jpg   # opcional (negativos)
```
- Pastas em `known/` definem as **classes (labels)**.
- Em `test/`, a pasta `unknown/` (se existir) é tratada como negativos.

---

## 🚀 Como Usar

### 1. Ambiente Virtual & Dependências

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Executando o Coletor

```bash
python gqm_runner.py \
  --repo-path /path/to/source_code_final \
  --known-dir /path/to/known \
  --test-dir  /path/to/test \
  --out-dir   ./gqm_out \
  --model hog \
  --tolerance 0.6
```
- `--repo-path`: diretório do código a ser inspecionado (ex.: `source_code_final`)
- `--known-dir`: base de rostos rotulados
- `--test-dir`: base de testes

### 3. Saídas Geradas

| Arquivo                       | Descrição                                      |
|-------------------------------|------------------------------------------------|
| `gqm_out/gqm_report.json`     | Relatório completo (métricas G1, G2, G3, G4)   |
| `gqm_out/predictions.csv`     | Pares (y_true, y_pred) por imagem              |

---

## ⚙️ Parâmetros Importantes

- Para **CNN mais precisa**, use `--model cnn` (exige GPU/CUDA).
- Ajuste `--tolerance` (`0.4–0.6` recomendado; menor = mais estrito).
- Se não houver testes, cobertura será apenas informativa.
- Para datasets grandes, aumente `--resource-sample-interval`.

---

## 🧩 Detecção Automática de Condições

O `gqm_runner.py` detecta condições nos nomes dos arquivos de teste usando sufixos como:

```
_lowlight, _backlight, _blur, _occluded, _noisy, _angle15, _angle30, _angle45, _lowres, _highres
```

**Exemplo de nome:**  
`alice_01_lowlight.jpg` ou `bob_03_angle45.png`

---
