# Model Card - Churn Prediction MLP

## 1. Nome do modelo
pytorch_mlp


## 2. Versão
1.0.0


## 3. Objetivo

O modelo tem como objetivo prever a probabilidade de churn de clientes de telecomunicações.
Churn representa o cancelamento do serviço pelo cliente. A previsão pode apoiar ações de retenção, priorizando clientes com maior risco de saída.


## 4. Tipo de problema

Classificação binária supervisionada.
Classes:

| Classe | Significado          |
|   0    | Cliente não cancelou |
|   1    | Cliente cancelou     |


## 5. Dataset utilizado

Foi utilizado o dataset Telco Customer Churn.
A variável alvo é `Churn`.
A coluna `customerID` foi removida por ser um identificador e não representar comportamento útil para o modelo.
A coluna `TotalCharges` foi convertida para valor numérico e valores ausentes foram preenchidos com a mediana.


## 6. Features utilizadas

O modelo utiliza variáveis cadastrais, contratuais e de consumo, como:

* gênero;
* idade/senioridade;
* tempo como cliente;
* serviços contratados;
* tipo de contrato;
* método de pagamento;
* cobrança mensal;
* cobrança total.


## 7. Pré-processamento

O pré-processamento foi feito com Scikit-Learn.

Foram aplicadas as seguintes etapas:

| Tipo de variável  | Transformação                                           |
|     Numérica      | Imputação com mediana e padronização com StandardScaler |
|    Categórica     | Imputação com valor mais frequente e OneHotEncoder      |

O preprocessor treinado foi salvo em:
models/preprocessor.joblib


## 8. Algoritmo principal

O modelo principal é uma rede neural MLP implementada em PyTorch.

Arquitetura:
Input
↓
Linear(input_size, 64)
↓
ReLU
↓
Dropout(0.30)
↓
Linear(64, 32)
↓
ReLU
↓
Dropout(0.20)
↓
Linear(32, 1)

A saída do modelo é convertida em probabilidade usando sigmoid.


## 9. Estratégia de treinamento

| Parâmetro        |             Valor |
| ---------------- | ----------------- |
| Batch size       |                64 |
| Learning rate    |             0.001 |
| Máximo de épocas |               100 |
| Early stopping   |               Sim |
| Patience         |                10 |
| Loss function    | BCEWithLogitsLoss |
| Otimizador       |              Adam |
| Seed             |                42 |

Observação: Além do split treino/teste estratificado, foi executada validação cruzada estratificada com `StratifiedKFold` em 5 folds para os modelos baseline. Essa etapa foi utilizada para avaliar a estabilidade dos modelos em diferentes divisões da base, preservando a proporção entre clientes churn e não churn em cada fold.


## 10. Métricas da MLP

| Métrica   |  Valor |
| --------- | -----: |
| Accuracy  | 0.7899 |
| Precision | 0.6283 |
| Recall    | 0.5107 |
| F1-score  | 0.5634 |
| ROC-AUC   | 0.8401 |
| PR-AUC    | 0.6327 |


## 11. Comparação com baselines

| Modelo              | ROC-AUC |
| ------------------- | ------- |
| Gradient Boosting   |  0.8432 |
| Logistic Regression |  0.8419 |
| MLP PyTorch         |  0.8401 |
| Random Forest       |  0.8187 |
| Dummy Classifier    |  0.5000 |

A MLP apresentou desempenho próximo aos melhores baselines. Já o Gradient Boosting teve o maior ROC-AUC, mas a MLP cumpriu o objetivo desse desafio por implementar uma rede neural funcional com PyTorch.


## 12. Interpretação de saída

| Campo               | Descrição                                              |
| ------------------- | ------------------------------------------------------ |
| `churn_prediction`  | Classe prevista, sendo 0 para não churn e 1 para churn |
| `churn_probability` | Probabilidade estimada de churn                        |
| `risk_level`        | Classificação de risco: baixo, medio ou alto           |
| `model_name`        | Nome do modelo                                         |
| `model_version`     | Versão do modelo                                       |

| Probabilidade         | Risco |
| --------------------- | ----- |
| Menor que 0.40        | baixo |
| Entre 0.40 e 0.69     | medio |
| Maior ou igual a 0.70 | alto  |


## 13. Uso recomendado

O modelo pode ser usado para apoiar times de retenção de clientes, priorizando clientes com maior risco de churn.

Exemplo de uso:

* identificar clientes de risco;
* priorizar campanhas de retenção;
* apoiar ações de relacionamento;
* monitorar segmentos com maior tendência de cancelamento;


## 14. Uso não recomendado

O modelo não deve ser usado como única fonte de decisão para:

* cancelar contratos;
* negar atendimento;
* aplicar penalidades;
* tomar decisões sensíveis sem revisão humana;


## 15. Limitações

* O modelo foi treinado com um dataset histórico e pode não refletir mudanças recentes do mercado.
* A MLP não superou claramente o Gradient Boosting e a Regressão Logística.
* O dataset pode conter vieses relacionados ao perfil dos clientes coletados.
* O modelo ainda não possui explicabilidade avançada.
* O deploy em produção não foi implementado nesta versão.
* Não há monitoramento real de drift nesta etapa.


## 16. Riscos

Possíveis riscos:

* falso positivo: cliente é classificado como churn, mas não cancelaria;
* falso negativo: cliente cancelaria, mas o modelo não identifica;
* uso do modelo sem análise humana;
* degradação de performance ao longo do tempo;

No contexto de churn, falsos negativos podem ser especialmente relevantes, porque representam clientes que poderiam ter recebido ações de retenção, mas não foram identificados.


## 17. Monitoramento recomendado

Em produção, recomenda-se monitorar:

* volume de requisições;
* latência da API;
* taxa de erro;
* distribuição das probabilidades;
* proporção de clientes classificados como alto risco;
* data drift;
* model drift;
* métricas reais após obtenção do churn observado;


## 18. Responsável pela solução

Projeto acadêmico desenvolvido para o Tech Challenge 01 FIAP 9MLET
