# Aspectos Gerais do Trabalho

---

## 1. Contexto do Software

O projeto é um exemplo de aplicação de reconhecimento facial em Python, usando bibliotecas como `face_recognition` e `OpenCV`, com funcionalidades para detectar rostos em imagens e reconhecê-los com base em um dataset de treinamento.

---

## 2. Metas de Medição (Goals)

As metas definem o que queremos avaliar no software:

- **G1:** Avaliar a Correção do reconhecimento facial.
- **G2:** Medir a eficiência de desempenho da aplicação (tempo de resposta).
- **G3:** Avaliar a facilidade de manutenção e qualidade do código.
- **G4:** Garantir a confiabilidade da aplicação em diferentes cenários (luz, ângulo, qualidade da imagem).

---

## 3. Questões (Questions)

As questões direcionam a coleta de métricas para cada meta.

### Para G1 (Correção)
- **Q1:** Qual a taxa de acerto do reconhecimento facial?
- **Q2:** Qual a taxa de falsos positivos (reconhecer uma pessoa errada)?

### Para G2 (desempenho)
- **Q3:** Quanto tempo o software leva para detectar e reconhecer rostos em uma imagem?
- **Q4:** Qual o uso de CPU/memória durante a execução?

### Para G3 (qualidade do código)
- **Q5:** Qual a complexidade ciclomática média das funções?
- **Q6:** Qual a cobertura de testes unitários?

### Para G4 (confiabilidade)
- **Q7:** Qual a variação da taxa de acerto em diferentes condições (luz, ângulo, resolução)?
- **Q8:** O sistema mantém desempenho consistente em lotes de imagens maiores (stress test)?

---

## 4. Definição das Métricas

| Meta         | Métrica                                      | Descrição                                              |
|--------------|----------------------------------------------|--------------------------------------------------------|
| Correção    | **M1:** Accuracy                             | (nº de rostos corretamente reconhecidos / total de rostos) × 100 |
|              | **M2:** False Positive Rate (FPR)            |                                                        |
| Desempenho   | **M3:** Tempo médio de processamento (ms)     |                                                        |
|              | **M4:** Uso médio de CPU e memória (%)       |                                                        |
| Qualidade    | **M5:** Complexidade ciclomática             | (ex. com radon)                                        |
| do código    | **M6:** Cobertura de testes unitários (%)     | (ex. com pytest --cov)                                 |
| Confiabilidade| **M7:** Accuracy por condição de teste       | (dataset em diferentes iluminações/ângulos)            |
|              | **M8:** Tempo médio de resposta em batch      |                                                        |

---

## 5. Extração das Métricas

**Ferramentas sugeridas:**
- **M1, M2, M7:** Dataset com imagens rotuladas + `scikit-learn` (`classification_report`)
- **M3, M8:** `time.perf_counter()` + testes de stress em batch
- **M4:** `psutil` para uso de recursos
- **M5:** `radon cc` para complexidade
- **M6:** `pytest --cov` para cobertura de testes

---

## 6. Resultados Esperados

- **Correção:** Espera-se accuracy > 90% em condições ideais, mas queda significativa em imagens com baixa iluminação.
- **Desempenho:** Reconhecimento em imagens individuais em < 200ms, podendo subir para ~1s em lotes grandes.
- **Qualidade do código:** Complexidade ciclomática média baixa (< 10), porém pouca cobertura de testes.
- **Confiabilidade:** Sistema funciona bem com imagens frontais, mas falha em ângulos > 45°.