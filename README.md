# Tech Challenge 01 - Churn Prediction com MLP

Projeto desenvolvido para o Tech Challenge da Pós Tech FIAP em Machine Learning Engineering.


## 1. Contexto do problema

Uma empresa de telecomunicações deseja reduzir a perda de clientes, conhecida como churn.

Churn acontece quando um cliente deixa de usar o serviço da empresa. Em empresas de telecom, esse problema é relevante porque a aquisição de novos clientes costuma ser mais cara do que a retenção de clientes atuais.

O objetivo deste projeto é construir uma solução de Machine Learning capaz de prever a probabilidade de um cliente cancelar o serviço, permitindo que a empresa priorize ações de retenção.


## 2. Objetivo técnico

Construir um pipeline de Machine Learning para classificação binária, utilizando:

- análise exploratória de dados;
- modelos baseline com Scikit-Learn;
- rede neural MLP com PyTorch;
- rastreamento de experimentos com MLflow;
- API de inferência com FastAPI;
- validação de entrada com Pydantic;
- testes automatizados com Pytest;
- linting com Ruff;
- documentação técnica do modelo e da solução.


## 3. Dataset

Foi utilizado o dataset público Telco Customer Churn.

Arquivo esperado:

data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

A variável alvo é:

Churn

Mapeamento da variável alvo:

No  -> 0
Yes -> 1

Principais variáveis utilizadas:

| Coluna          | Descrição                                   |
| --------------- | ------------------------------------------- |
| gender          | Gênero do cliente                           |
| SeniorCitizen   | Indica se o cliente é idoso                 |
| Partner         | Indica se o cliente possui parceiro/cônjuge |
| Dependents      | Indica se possui dependentes                |
| tenure          | Tempo como cliente                          |
| PhoneService    | Possui serviço de telefone                  |
| InternetService | Tipo de serviço de internet                 |
| Contract        | Tipo de contrato                            |
| PaymentMethod   | Método de pagamento                         |
| MonthlyCharges  | Valor mensal cobrado                        |
| TotalCharges    | Valor total cobrado                         |
| Churn           | Indica se houve cancelamento                |


## ## 4. Estrutura do projeto

```text
tech-challenge-churn-mlp/
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/
│       └── .gitkeep
├── docs/
│   ├── architecture.md
│   ├── ml_canvas.md
│   ├── model_card.md
│   └── monitoring_plan.md
├── models/
│   ├── baseline_results.json
│   ├── mlp_metadata.json
│   ├── mlp_model.pt
│   ├── preprocessor.joblib
│   └── stratified_cv_results.json
├── notebooks/
│   ├── 01_eda_telco_churn.ipynb
│   └── README.md
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── api.py
│       ├── config.py
│       ├── data.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── stratified_cv.py
│       ├── train_baselines.py
│       └── train_mlp.py
├── tests/
│   ├── test_api.py
│   ├── test_schema.py
│   └── test_smoke.py
├── .gitignore
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```


## 5. Tecnologias utilizadas

| Tecnologia   | Uso no projeto                          |
| ------------ | --------------------------------------- |
| Python       | Linguagem principal                     |
| Pandas       | Manipulação dos dados                   |
| NumPy        | Operações numéricas                     |
| Scikit-Learn | Pré-processamento, baselines e métricas |
| PyTorch      | Treinamento da rede neural MLP          |
| MLflow       | Rastreamento de experimentos            |
| FastAPI      | Criação da API de inferência            |
| Pydantic     | Validação dos dados de entrada          |
| Uvicorn      | Servidor local da API                   |
| Pytest       | Testes automatizados                    |
| Ruff         | Verificação de qualidade do código      |
| Joblib       | Salvamento do preprocessor              |


## 6. Como configurar o ambiente

* 6.1 Criar ambiente virtual - Execute:
python -m venv .venv

* 6.2 Ativar ambiente virtual no Windows - Execute:
.venv\Scripts\activate

* 6.3 Instalar dependências - Execute:
pip install -r requirements.txt


## 7. Análise Exploratória dos Dados

A análise exploratória foi registrada no notebook:

```text
notebooks/01_eda_telco_churn.ipynb
```

O notebook contém:

* volume do dataset;
* tipos de dados;
* verificação de valores ausentes;
* distribuição da variável alvo `Churn`;
* análise de variáveis numéricas;
* análise de churn por tipo de contrato;
* análise de churn por método de pagamento;
* principais conclusões para a modelagem.

Pra executar o notebook, abra o arquivo no VS Code ou Jupyter e selecione o kernel Python 3.11.


## 8.ML Canvas

O ML Canvas do projeto está documentado em:

```text
docs/ml_canvas.md
```
Esse documento resume o problema de negócio, stakeholders, métricas técnicas, métrica de negócio, hipóteses iniciais, riscos e critério de sucesso da solução.


## 9. Como treinar os modelos baseline
Execute:
python -m src.churn.train_baselines

Esse comando acima treina:
* DummyClassifier;
* Logistic Regression;
* Random Forest;
* Gradient Boosting;

Os resultados são salvos em:
`models/baseline_results.json`


## 10. Como treinar a MLP em PyTorch
Execute:
python -m src.churn.train_mlp

Esse comando treina a rede neural MLP, aplica early stopping, registra métricas no MLflow e salva os artefatos:
* models/mlp_model.pt
* models/preprocessor.joblib
* models/mlp_metadata.json


## 11. Resultados dos modelos

* 9.1 Baselines

| Modelo              | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: | -----: |
| Gradient Boosting   |   0.8062 |    0.6735 | 0.5241 | 0.5895 |  0.8432 | 0.6642 |
| Logistic Regression |   0.8055 |    0.6572 | 0.5588 | 0.6040 |  0.8419 | 0.6334 |
| Random Forest       |   0.7750 |    0.5979 | 0.4652 | 0.5233 |  0.8187 | 0.6075 |
| Dummy Classifier    |   0.7346 |    0.0000 | 0.0000 | 0.0000 |  0.5000 | 0.2654 |

* 9.2 MLP PyTorch

| Modelo      | Accuracy | Precision | Recall |     F1 | ROC-AUC | PR-AUC |
| ----------- | -------: | --------: | -----: | -----: | ------: | -----: |
| MLP PyTorch |   0.7899 |    0.6283 | 0.5107 | 0.5634 |  0.8401 | 0.6327 |

Observação: Também foi executada validação cruzada estratificada com `StratifiedKFold` em 5 folds para os modelos baseline, com os resultados salvos em `models/stratified_cv_results.json` e registrados no MLflow.


## 12. Interpretação dos resultados

O modelo DummyClassifier teve ROC-AUC de 0.50, funcionando como referência mínima.

Os modelos Logistic Regression, Gradient Boosting e MLP tiveram desempenho parecido, com ROC-AUC acima de 0.84 para os dois melhores baselines e aproximadamente 0.84 para a MLP.

A MLP PyTorch cumpriu o objetivo principal do desafio ao demonstrar uma rede neural funcional para classificação binária, com batching, early stopping, métricas, salvamento do modelo e rastreamento de experimentos.

Em dados tabulares, é comum que modelos como Gradient Boosting ou Logistic Regression sejam competitivos ou até superiores a redes neurais simples. Por isso, a comparação com baselines é essencial para avaliar se a complexidade adicional da rede neural é justificada.


## 13. Como visualizar experimentos no MLflow
Executar:
mlflow ui

Depois acesse no navegador:
http://127.0.0.1:5000

Os experimentos registrados incluem:

* baselines;
* MLP PyTorch;
* parâmetros;
* métricas;
* artefatos;


## 14. Como rodar a API

Execute:
uvicorn src.churn.api:app --reload

Depois acesse:
http://127.0.0.1:8000/docs


## 15. Endpoints da API

| Método | Endpoint   | Descrição                                                   |
| ------ | ---------- | ----------------------------------------------------------- |
| GET    | `/`        | Informações básicas da API                                  |
| GET    | `/health`  | Verifica se a API está saudável e se o modelo foi carregado |
| POST   | `/predict` | Recebe dados de um cliente e retorna a previsão de churn    |


## 16. Exemplo de requisição para /predict (JSON)

{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}


## 17. Exemplo de resposta da API (JSON)


{
  "churn_prediction": 1,
  "churn_probability": 0.6846,
  "risk_level": "medio",
  "model_name": "pytorch_mlp",
  "model_version": "1.0.0"
}


Interpretação:

churn_prediction = 1: cliente classificado como provável churn;
churn_probability = 0.6846: probabilidade estimada de churn de 68,46%;
risk_level = medio: risco classificado como médio;


## 18. Como rodar os testes
Execute:
pytest

Resultado esperado:
5 passed


## 19. Como rodar o lint
Execute:
ruff check .

Resultado esperado:
All checks passed!


## 20. Reprodutibilidade

O projeto utiliza:

* ambiente virtual
* dependências listadas em requirements.txt
* configuração em pyproject.toml
* seed fixa
* artefatos salvos em models/
* código modular em src/churn/
* split treino/teste estratificado
* validação cruzada estratificada com StratifiedKFold em 5 folds para os modelos baseline


## 21. Limitações

Algumas limitações do projeto:

* o dataset é histórico e pode não representar clientes atuais;
* a base pode conter viés relacionado ao perfil dos clientes presentes no dataset;
* a MLP não superou claramente os melhores baselines;
* não foi realizado tuning avançado de hiperparâmetros;
* a API está executando localmente;
* o deploy em nuvem não foi implementado nesta versão;


## 22. Próximos passos

Possíveis melhorias futuras:

* realizar tuning de hiperparâmetros;
* testar novas arquiteturas de rede neural;
* aplicar validação cruzada estratificada completa;
* incluir explicações com feature importance ou SHAP;
* criar deploy em nuvem;
* adicionar autenticação na API;
* monitorar data drift e model drift em produção;
* criar rotina de retreino periódico;