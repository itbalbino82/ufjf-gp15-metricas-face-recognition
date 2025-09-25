1. Contexto do Software

O projeto é um exemplo de aplicação de reconhecimento facial em Python, usando bibliotecas como face_recognition e OpenCV, com funcionalidades para detectar rostos em imagens e reconhecê-los com base em um dataset de treinamento.

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

2. Metas de Medição (Goals)

As metas definem o que queremos avaliar no software. Exemplos:

G1: Avaliar a corretude do reconhecimento facial.

G2: Medir a eficiência de desempenho da aplicação (tempo de resposta).

G3: Avaliar a facilidade de manutenção e qualidade do código.

G4: Garantir a confiabilidade da aplicação em diferentes cenários (luz, ângulo, qualidade da imagem).

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

3. Questões (Questions)

As questões direcionam a coleta de métricas para cada meta.
-------------------------------------------------------------------------------------------------------------------------
Para G1 (corretude)

Q1: Qual a taxa de acerto do reconhecimento facial?

Q2: Qual a taxa de falsos positivos (reconhecer uma pessoa errada)?
-------------------------------------------------------------------------------------------------------------------------
Para G2 (desempenho)

Q3: Quanto tempo o software leva para detectar e reconhecer rostos em uma imagem?

Q4: Qual o uso de CPU/memória durante a execução?
-------------------------------------------------------------------------------------------------------------------------
Para G3 (qualidade do código)

Q5: Qual a complexidade ciclomática média das funções?

Q6: Qual a cobertura de testes unitários?
-------------------------------------------------------------------------------------------------------------------------
Para G4 (confiabilidade)

Q7: Qual a variação da taxa de acerto em diferentes condições (luz, ângulo, resolução)?

Q8: O sistema mantém desempenho consistente em lotes de imagens maiores (stress test)?

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

4. Definição das Métricas

As métricas (M) são extraídas a partir das perguntas.
-------------------------------------------------------------------------------------------------------------------------
Corretude (G1)

M1: Accuracy = (nº de rostos corretamente reconhecidos / total de rostos) × 100

M2: False Positive Rate (FPR)
-------------------------------------------------------------------------------------------------------------------------
Desempenho (G2)

M3: Tempo médio de processamento por imagem (ms)

M4: Uso médio de CPU e memória (%)
-------------------------------------------------------------------------------------------------------------------------
Qualidade do código (G3)

M5: Complexidade ciclomática (ex. com radon)

M6: Cobertura de testes unitários (%) (ex. com pytest --cov)
-------------------------------------------------------------------------------------------------------------------------
Confiabilidade (G4)

M7: Accuracy por condição de teste (dataset em diferentes iluminações/ângulos)

M8: Tempo médio de resposta em batch de N imagens

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

5. Extração das Métricas

Sugestão de ferramentas para coleta:

M1, M2, M7: Criar dataset com imagens rotuladas → usar scikit-learn (classification_report) para medir accuracy, precision, recall, FPR.

M3, M8: Medir tempo com time.perf_counter() e testes de stress com batch de imagens.

M4: Medir uso de recursos com psutil.

M5: Usar radon cc para medir complexidade.

M6: Usar pytest --cov para extrair cobertura de testes.

-------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------

6. Resultados esperados

Corretude: Espera-se accuracy > 90% em condições ideais, mas queda significativa em imagens com baixa iluminação.

Desempenho: Reconhecimento em imagens individuais em < 200ms, mas tempo pode subir para ~1s em lotes grandes.

Qualidade do código: Complexidade ciclomática média baixa (< 10), porém pouca cobertura de testes (o repositório de exemplo não tem testes extensivos).

Confiabilidade: Sistema funciona bem com imagens frontais, mas falha em ângulos > 45°.