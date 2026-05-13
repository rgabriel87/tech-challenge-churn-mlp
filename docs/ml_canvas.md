# ML Canvas - Churn Prediction

## 1. Problema de negócio

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. O cancelamento de clientes, conhecido como churn, impacta diretamente receita, relacionamento e previsibilidade do negócio.
O objetivo é antecipar quais clientes possuem maior risco de cancelamento para apoiar ações preventivas de retenção.


## 2. Objetivo do modelo

Construir um modelo de classificação binária capaz de estimar a probabilidade de churn de um cliente.
A saída esperada do modelo é:

* classe prevista: churn ou não churn;
* probabilidade de churn;
* nível de risco: baixo, médio ou alto.


## 3. Stakeholders

| Stakeholder              | Interesse                                    |
|--------------------------|----------------------------------------------|
| Diretoria                | Reduzir perda de clientes e proteger receita |
| Time de retenção         | Priorizar clientes com maior risco           |
| Time de marketing        | Apoiar campanhas segmentadas                 |
| Time de dados/tecnologia | Manter pipeline, modelo e API                |
| Atendimento ao cliente   | Apoiar ações preventivas de relacionamento   |


## 4. Métrica de negócio

A métrica de negócio sugerida é o custo de churn evitado.
Exemplo:
Se clientes de alto risco forem identificados com antecedência, a empresa pode direcionar ações de retenção para reduzir cancelamentos e preservar sua receita.


## 5. Métricas técnicas

As métricas técnicas utilizadas foram:

| Métrica   | Motivo                                           |
|-----------|--------------------------------------------------|
| Accuracy  | Mede acerto geral                                |
| Precision | Mede qualidade dos alertas positivos             |
| Recall    | Mede capacidade de encontrar clientes churn      |
| F1-score  | Equilibra precision e recall                     |
| ROC-AUC   | Mede capacidade geral de separação entre classes |
| PR-AUC    | Útil quando a classe positiva é menos frequente  |

A métrica principal considerada foi ROC-AUC, pois o problema envolve classificação binária e comparação da capacidade de separação entre clientes churn e não churn.


## 6. Variável alvo

A variável alvo é `Churn`.
Mapeamento:

| Valor original | Valor numérico |
|----------------|----------------|
| No             | 0              |
| Yes            | 1              |


## 7. Dados utilizados

Foi utilizado o dataset público Telco Customer Churn.
O dataset contém variáveis cadastrais, contratuais e de consumo, como:

* tempo como cliente;
* tipo de contrato;
* método de pagamento;
* serviços contratados;
* cobrança mensal;
* cobrança total;


## 8. Hipóteses iniciais

Algumas hipóteses avaliadas na EDA:
1. Clientes com contrato mensal podem ter maior risco de churn.
2. Clientes com determinados métodos de pagamento podem apresentar maior taxa de churn.
3. Clientes com menor tempo de relacionamento podem ter maior tendência de fazer o cancelamento.
4. Valores mensais mais altos podem influenciar no risco de cancelamento.


## 9. Restrições

* O dataset é histórico e pode não refletir mudanças recentes do mercado.
* A solução atual roda localmente.
* O deploy em nuvem foi tratado como melhoria futura.
* O modelo não deve ser usado como única fonte de decisão sem análise humana.


## 10. Riscos de erro

| Tipo de erro   | Impacto                                                        |
|----------------|----------------------------------------------------------------|
| Falso positivo | Cliente é classificado como risco de churn, mas não cancelaria |
| Falso negativo | Cliente com risco real de churn não é identificado             |

No contexto de churn, falsos negativos podem ser críticos, pois representam clientes que poderiam receber ações preventivas, mas não foram priorizados.


## 11. Solução proposta

A solução proposta é um pipeline end-to-end com:

* EDA em notebook;
* pré-processamento com Scikit-Learn;
* modelos baseline;
* MLP em PyTorch;
* rastreamento com MLflow;
* API FastAPI;
* testes automatizados;
* documentação técnica;
* Model Card;
* plano de monitoramento;


## 12. Critério de sucesso

O projeto é considerado bem-sucedido se:

* a MLP for treinada e comparada com baselines;
* os experimentos forem registrados no MLflow;
* a API `/predict` funcionar;
* os testes passarem;
* o código estiver organizado;
* o README e a documentação permitirem reproduzir a solução;