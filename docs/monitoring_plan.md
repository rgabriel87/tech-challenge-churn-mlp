# Plano de Monitoramento

## 1. Objetivo

Esse documento descreve como a solução de previsão de churn deveria ser monitorada em um cenário de produção.
O monitoramento é necessário porque modelos de Machine Learning podem perder performance ao longo do tempo, principalmente quando o comportamento dos clientes muda.


## 2. O que monitorar na API

| Métrica                | Por que monitorar?                   |
|------------------------|--------------------------------------|
| Volume de requisições  | Entender uso da API                  |
| Latência               | Garantir tempo de resposta adequado  |
| Taxa de erro           | Identificar falhas no serviço        |
| Status do modelo       | Verificar se o modelo está carregado |
| Tempo de processamento | Medir performance da inferência      |

A API já retorna o header:
X-Process-Time

* Esse header indica o tempo de processamento da requisição.


## 3. O que monitorar no modelo

| Métrica                          | Descrição                                                                  |
| -------------------------------- | -------------------------------------------------------------------------- |
| Distribuição das probabilidades  | Verificar se o modelo está retornando valores muito diferentes do esperado |
| Proporção de clientes alto risco | Detectar mudanças no perfil das previsões                                  |
| Taxa real de churn               | Comparar previsão com resultado observado                                  |
| Precision                        | Verificar qualidade dos alertas positivos                                  |
| Recall                           | Verificar capacidade de capturar clientes que realmente cancelam           |
| ROC-AUC                          | Monitorar capacidade geral de separação entre churn e não churn            |
| PR-AUC                           | Monitorar performance em cenário de classe positiva menos frequente        |


## 4. Data drift

Data drift ocorre quando a distribuição dos dados de entrada muda ao longo do tempo. Exemplos:

* mudança no perfil dos clientes;
* novos tipos de contrato;
* alteração nos preços;
* mudança nos métodos de pagamento;
* novos serviços oferecidos pela empresa.

# Variáveis recomendadas para monitoramento:

* tenure;
* MonthlyCharges;
* TotalCharges;
* Contract;
* PaymentMethod;
* InternetService;


## 5. Model drift

Model drift ocorre quando a relação entre as variáveis de entrada e o churn muda.

Exemplo:
Um tipo de contrato que antes indicava alto risco de churn pode deixar de ter esse comportamento no futuro.
Pra identificar model drift, é necessário comparar as previsões com o churn real observado posteriormente.


## 6. Logs recomendados

Em produção, cada requisição poderia registrar:

* data e hora;
* versão do modelo;
* probabilidade prevista;
* classe prevista;
* tempo de processamento;
* status da requisição;
* identificador anônimo do cliente;
* registrar erro, caso ocorra;

Observação: Não se recomenda registrar dados sensíveis diretamente nos logs.


## 7. Alertas recomendados
Possíveis alertas:

| Situação                               | Ação                                           |
| -------------------------------------- | ---------------------------------------------- |
| Latência acima do limite               | Verificar infraestrutura/API                   |
| Aumento de erros 500                   | Investigar falha no serviço                    |
| Queda brusca no volume de requisições  | Verificar integração com sistemas consumidores |
| Aumento extremo de clientes alto risco | Investigar mudança nos dados                   |
| Queda de performance real              | Avaliar retreino do modelo                     |


## 8. Estratégia de retreino
Sugestão de retreino:

* retreino mensal ou trimestral;
* retreino quando houver queda relevante nas métricas;
* retreino quando houver mudança importante nos produtos, preços ou contratos;
* retreino após acúmulo de novos dados rotulados;

Fluxo sugerido:
Coleta de novos dados
↓
Validação de qualidade
↓
Treinamento de novos candidatos
↓
Comparação com modelo atual
↓
Aprovação técnica
↓
Publicação de nova versão
↓
Monitoramento pós-deploy


## 9. Critérios para nova versão

Uma nova versão do modelo só deveria substituir a atual se:

* apresentar performance igual ou superior nas métricas principais;
* não aumentar riscos de negócio;
* passar nos testes automatizados;
* manter compatibilidade com a API;
* ser documentada com nova versão de Model Card;


## 10. Considerações finais

O modelo atual é adequado para uma prova de conceito acadêmica e para demonstração de uma solução de Machine Learning Engineering.

Para uso real em produção, seria necessário adicionar:

* autenticação;
* monitoramento centralizado;
* banco de logs;
* versionamento formal de modelos;
* pipeline automatizado de retreino;
* testes de carga;
* observabilidade de dados e modelo;