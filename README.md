# Análise de Sentimentos com Machine Learning

Projeto de classificação de sentimentos (positivo / negativo) em reviews de produtos, utilizando **TF-IDF** + **Regressão Logística**, com pipeline completo de pré-processamento, otimização de hiperparâmetros via **GridSearchCV** e persistência do modelo treinado para deploy.

---

## Sobre o Projeto

O modelo recebe um texto (review de cliente) e retorna a previsão do sentimento associado: **Positivo** ou **Negativo**. Toda a etapa de limpeza, vetorização e classificação é encapsulada em um `Pipeline` do scikit-learn, o que garante que o mesmo tratamento aplicado no treino seja aplicado em produção.

## Tecnologias Utilizadas

- Python 3
- pandas / numpy
- scikit-learn (Pipeline, TfidfVectorizer, LogisticRegression, GridSearchCV)
- seaborn / matplotlib
- joblib (serialização do modelo)

## Estrutura

```
ML_AnalisedeSentimento/
├── ML.ModeloSentimento.py       # Script principal (treino + deploy)
├── dataset.csv                  # Base de reviews rotulados
├── modelo_sentimento_v1.joblib  # Modelo treinado salvo
└── README.md
```

## Pipeline de Processamento

1. **Carga e EDA** — leitura do CSV, verificação de nulos e distribuição das classes.
2. **Limpeza de texto** — conversão para minúsculas, remoção de acentos, pontuação e espaços extras.
3. **Engenharia de atributos** — mapeamento `positivo → 1` / `negativo → 0`.
4. **Divisão treino/teste** — `train_test_split` com estratificação (75% / 25%).
5. **Pipeline de modelagem** — `TfidfVectorizer` + `StandardScaler` + `LogisticRegression`.
6. **Otimização** — `GridSearchCV` com validação cruzada (5-fold) sobre `max_features`, `ngram_range`, `C`, `penalty` e `max_iter`.
7. **Avaliação** — acurácia, classification report e matriz de confusão.
8. **Deploy** — modelo serializado com `joblib` e recarregado para classificar novos reviews.

## Como Executar

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib
python ML.ModeloSentimento.py
```

## Exemplo de Uso

```python
import joblib

modelo = joblib.load('modelo_sentimento_v1.joblib')
reviews = ["O produto chegou rápido e funciona perfeitamente."]
print(modelo.predict(reviews))   # [1] -> Positivo
```

## Resultados

O modelo gera ao final do treinamento:
- Melhores hiperparâmetros encontrados pelo GridSearch
- Acurácia no conjunto de teste
- Relatório de precisão / recall / f1-score por classe
- Matriz de confusão visualizada com seaborn

---

Projeto desenvolvido como prática do curso **Python - Do Básico à Aplicação de IA**.
