# EXEMPLO DE RELATÓRIO COM A PRÁTICA GQM APLICADA AO SOFTWARE — GQM Face Recognition App

Este documento resume as métricas extraídas com base na metodologia **Goal–Question–Metric (GQM)** aplicada ao projeto *face-recognition*.

---

## 📌 G1 — Corretude
- **Accuracy geral:** 92.4 %  
- **Precision média:** 91.0 %  
- **Recall médio:** 90.5 %  
- **F1-score médio:** 90.7 %  
- **Taxa de falsos positivos (FPR):** 4.3 %  

👉 O sistema apresenta boa assertividade em condições normais de teste, com baixa incidência de falsos positivos.

---

## ⚡ G2 — Desempenho
- **Tempo médio por imagem:** 185 ms  
- **p95 latência:** 320 ms  
- **Throughput:** ~5.4 imagens/s  
- **Tempo total de execução (100 imagens):** 18.5 s  

👉 O desempenho é adequado para uso offline/lote. Para cenários em tempo real, otimização ou GPU pode ser necessária.

---

## 🖥️ G2 — Recursos
- **Uso médio de CPU:** 58 %  
- **CPU máxima observada:** 86 %  
- **Memória média:** 220 MB  
- **Memória máxima:** 310 MB  

👉 O consumo de recursos é moderado, viável em máquinas comuns com pelo menos 4 cores e 4 GB de RAM.

---

## 🛠️ G3 — Qualidade de Código
- **Complexidade ciclomática (Radon):**
  - Rank A: 18 funções
  - Rank B: 6 funções
  - Rank C+: 2 funções
- **Pior caso:** função `process_frame` com complexidade 11 (Rank C)
- **Cobertura de testes:** 15 % (parcial; não há suíte extensa de testes no repo)  

👉 O código é simples na maioria das funções, mas carece de testes automatizados.

---

## 🔒 G4 — Confiabilidade
- **Accuracy (condições normais):** 92 %  
- **Accuracy (baixa iluminação):** 78 %  
- **Accuracy (ângulo > 45°):** 65 %  

👉 O sistema é sensível a variações ambientais, especialmente ângulo e iluminação.

---

## 📊 Conclusão
- O modelo atinge boa corretude em condições normais.  
- Há queda perceptível em cenários adversos, sugerindo necessidade de **aumento de dataset** e **data augmentation**.  
- O desempenho é aceitável, mas para cenários em tempo real recomenda-se GPU ou uso do modelo **CNN** otimizado.  
- O código tem baixa complexidade, mas faltam **testes unitários** para maior confiabilidade em manutenção.

---

✅ **Próximos Passos Recomendados:**
1. Expandir dataset de testes (incluindo mais variações de ângulo/iluminação).  
2. Adicionar testes automatizados com `pytest` para elevar cobertura.  
3. Avaliar uso de GPU para processamento em tempo real.  
4. Monitorar métricas em execuções contínuas para validar robustez.
