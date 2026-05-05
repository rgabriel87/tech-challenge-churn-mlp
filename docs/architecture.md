# Arquitetura da Solução

## 1. Visão geral

A solução foi construída para prever churn de clientes de telecomunicações usando um pipeline de Machine Learning Engineering.

O projeto cobre desde o carregamento dos dados até a exposição do modelo por uma API de inferência.

## 2. Fluxo geral

Dataset Telco Customer Churn
↓
Carregamento dos dados
↓
Limpeza e preparação
↓
Separação entre features e target
↓
Split treino/teste estratificado
↓
Pré-processamento com Scikit-Learn
↓
Treinamento dos baselines
↓
Treinamento da MLP em PyTorch
↓
Registro de experimentos no MLflow
↓
Salvamento dos artefatos
↓
API FastAPI carrega modelo e preprocessor
↓
Cliente envia JSON para /predict
↓
API retorna probabilidade de churn


## 3. Camadas da arquitetura

| Camada            | Responsabilidade                                      |
| ----------------- | ----------------------------------------------------- |
| Dados             | Armazenar dataset bruto e possíveis dados processados |
| Pré-processamento | Tratar variáveis numéricas e categóricas              |
| Modelagem         | Treinar baselines e rede neural MLP                   |
| Rastreamento      | Registrar experimentos no MLflow                      |
| Artefatos         | Salvar modelo, preprocessor e metadados               |
| API               | Expor o modelo para inferência                        |
| Testes            | Validar API, schema e funcionamento básico            |
| Documentação      | Explicar uso, arquitetura, métricas e as limitações   |


## 4. Organização do código
O código foi organizado como um pacote Python interno em src/churn

| Arquivo              | Função                                              |
| -------------------- | --------------------------------------------------- |
| `config.py`          | Centraliza aqui os caminhos e constantes do projeto |
| `data.py`            | Carrega e limpa o dataset                           |
| `preprocessing.py`   | Cria o pipeline de pré-processamento                |
| `train_baselines.py` | Treina os modelos baseline                          |
| `train_mlp.py`       | Treina a rede neural MLP com o PyTorch              |
| `predict.py`         | Carrega modelo salvo e executa predições            |
| `api.py`             | Cria a API FastAPI                                  |


## 5. Justificativa da arquitetura

A solução foi estruturada de forma modular para evitar um projeto baseado apenas em notebook.
Essa abordagem facilita:

* manutenção;
* testes automatizados;
* reaproveitamento de código;
* separação de responsabilidades;
* rastreabilidade dos experimentos;
* evolução futura para deploy em nuvem;


## 6. API de inferência
A API foi criada com FastAPI.

Endpoints disponíveis:
| Método | Endpoint   | Função                                         |
| ------ | ---------- | ---------------------------------------------- |
| GET    | `/`        | Retorna informações básicas                    |
| GET    | `/health`  | Verifica saúde da API e carregamento do modelo |
| POST   | `/predict` | Retorna previsão de churn                      |


## 7. Entrada da API

O endpoint /predict recebe um JSON com dados de um cliente. Exemplo:

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


## 8. Saída da API

Exemplo de resposta (JSON):

{
  "churn_prediction": 1,
  "churn_probability": 0.6846,
  "risk_level": "medio",
  "model_name": "pytorch_mlp",
  "model_version": "1.0.0"
}


## 9. Artefatos gerados

| Artefato                       | Descrição                     |
| ------------------------------ | ----------------------------- |
| `models/mlp_model.pt`          | Modelo MLP treinado           |
| `models/preprocessor.joblib`   | Pipeline de pré-processamento |
| `models/mlp_metadata.json`     | Metadados e métricas da MLP   |
| `models/baseline_results.json` | Resultados dos baselines      |


## 10. Rastreamento com MLflow
O MLflow foi utilizado para registrar:

* nome do modelo;
* parâmetros;
* métricas;
* artefatos;
* histórico de experimentos;


Pra visualizar, execute:
mlflow ui

Depois acesse:
http://127.0.0.1:5000


## 11. Estratégia de inferência

A API carrega o modelo e o preprocessor salvos em disco.
Para evitar recarregar os artefatos a cada requisição, foi utilizado cache com lru_cache.
Fluxo de inferência:

JSON de entrada
↓
Validação Pydantic
↓
Conversão para DataFrame
↓
Aplicação do preprocessor
↓
Conversão para tensor
↓
Inferência com MLP
↓
Sigmoid para probabilidade
↓
Resposta JSON


## 12. Possível arquitetura futura em nuvem

Uma evolução futura poderia usar:

Cliente / Sistema de CRM
↓
API Gateway ou Load Balancer
↓
Serviço FastAPI em container
↓
Modelo MLP carregado
↓
Logs e métricas em ferramenta de observabilidade
↓
Pipeline de retreino periódico

# Tecnologias possíveis:

* Docker;
* AWS App Runner;
* AWS ECS/Fargate;
* AWS CloudWatch;
* MLflow Tracking Server;
* banco de dados para logs de predição;