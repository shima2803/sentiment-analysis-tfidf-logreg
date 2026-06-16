# ============================================================
#Modelo de Classificação para Análise de Sentimentos
# ============================================================

# ---------- 2. Importação dos Pacotes ----------

import warnings
warnings.filterwarnings('ignore')
# Manipulação de dados e visualização
#import os
import re
import pandas as pd
import numpy as np
import unicodedata
import seaborn as sns
import matplotlib.pyplot as plt

# Pré-Processamento e Machine Learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

"""
ML com aprendizado supervisionado, onde receberemos X, e devolvera Y
"""
nome_arquivo = 'dataset.csv'

#carregar dataset
df_dsa = pd.read_csv(nome_arquivo)

#visualização dos dados
print(df_dsa.shape)

#primeiras linhas
print(df_dsa.head())

#EDA ( analise exploratória)
print(df_dsa.info())
print("\n Verificando valores ausentes: \n")
print(df_dsa.isnull().sum())

#Graf para amalise de positivo e negativo
print("\nDistribuição dos Sentimentos:\n")
sns.countplot( x = 'sentimento', data = df_dsa)
plt.title('Distribuição das Classes de Semtimento')
plt.show()

#Limpeza/tratamento dos dados
print(f"\nTamanho original do DataFreme: {len(df_dsa)}")
df_dsa.dropna(subset= ['texto_review'],inplace = True)
print(f"Tamanho do DataFrame após remover nulos: {len(df_dsa)}")

#limpar texto(acentos e escritas erradas)
def limpa_texto (texto):

    """
    Pipeline limpeza de texto:
    1. Converter para Minusculo
    2. Remover accentos e cedilha
    3. Romove pontuação e caracter especiais
    4. Remove espaços extra
    """

    if not isinstance(texto, str):
        return ""
    
    #Passo 1. Remover acentos
    # -> normalizar para formato NFKD
    texto_sem_acento = "".join(c for c in unicodedata.normalize('NFKD',texto) if unicodedata.category(c) != 'Mn')

    #Passo 2. Limpeza com Regex
    texto_limpo = texto_sem_acento.lower()

    texto_limpo = re.sub(r'[^a-z\s]','',texto_limpo)

    #remover espaço
    texto_limpo = re.sub(r'\s+',' ',texto_limpo).strip()

    return texto_limpo

df_dsa['texto_limpo'] = df_dsa['texto_review'].apply(limpa_texto)
#verificando limpeza
print(df_dsa.head())

# -------------------------------
# --- Engenharia de Atributos ---
#--------------------------------

df_dsa['sentimento_label'] = df_dsa['sentimento'].map({'positivo': 1,'negativo': 0})
print("\nDataFrame após a limpeza e mapeamento:\n")
print(df_dsa[['texto_limpo','sentimento_label']].head())


#Definir extrada X e saida Y
x = df_dsa['texto_limpo']
y = df_dsa['sentimento_label']

#Dividis os dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(x,y,test_size=0.25,random_state= 42,stratify= y)

#Pipeline de Modelagem Preditiva
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words = ['de','a','o','do','da','em','um'])),
    ('scaler', StandardScaler(with_mean = False)),
    ('logreg', LogisticRegression(solver = 'liblinear', random_state = 42, max_iter = 1000))
])


# Definir o grid de hiperparâmetros para otimização
parametros_grid = {
    'tfidf__max_features': [500, 1000, 2000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'logreg__C': [0.1, 1, 10],
    'logreg__penalty': ['l1', 'l2'],
    'logreg__max_iter': [5000, 6000]
}

# Configurar o GridSearchCV
grid_search = GridSearchCV(
    pipeline,              # Pipeline com as etapas de pré-processamento e modelo
    parametros_grid,       # Dicionário com as combinações de hiperparâmetros a serem testadas
    cv = 5,                # Número de divisões para validação cruzada (5-fold cross-validation)
    n_jobs = -1,           # Usa todos os núcleos disponíveis do processador para acelerar o processo
    scoring = 'accuracy',  # Métrica usada para avaliar o desempenho de cada combinação (aqui, acurácia)
    verbose = 1            # Nível de detalhamento do output durante a execução (1 exibe progresso básico)
)

#------------------------------
#--- Treinamento de Modelo ----
#------------------------------

print(f"\nTreinando de modelos com otimização de hiperparametro...")
grid_search.fit(X_treino,y_treino)

print("\nMelhor Hiperparametros encontrado:\n")
print(grid_search.best_params_)

#obter o melhor modelo
melhor_modelo = grid_search.best_estimator_
type(melhor_modelo)

#Avaliação do modelo e interpretação de Métricas
#previsao no conjunto de teste
y_pred = melhor_modelo.predict(X_teste)

#calculas as metricas de avaliação
acuracia = accuracy_score(y_teste,y_pred)

print(f"\nAcurácia do Modelo: {acuracia: .2%}\n")
print("\nRelatório de Classificação:\n")
report = classification_report(y_teste, y_pred, target_names=['Negativo','Positivo'])
print(report)

#visualizar matriz de confusão
cm = confusion_matrix(y_teste, y_pred)
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
            xticklabels = ['Negativo', 'Positivo'],
            yticklabels = ['Negativo', 'Positivo'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

#========================
#==== MLOps / Deploy ====
#========================

#se estivermos com a performace do modelo salvamos em disco
joblib.dump(melhor_modelo,'modelo_sentimento_v1.joblib')

#remove o modelo de treinamento
del melhor_modelo

modelo_deploy = joblib.load('modelo_sentimento_v1.joblib')
type(modelo_deploy)

novos_reviews = [
    "A bateria do celular não dura nada, péssima compra.",
    "Chegou antes do prazo e o produto é de ótima qualidade! Estou muito feliz.",
    "O serviço de atendimento foi rápido e eficiente.",
    "Não recomendo, veio faltando peças e a cor estava errada."
]

#função para prever sentimento
def prever_sentimento(review):

    """
    Recebe uma lista de textos de review e retorna a previsão de sentimento.
    O objeto 'melhor_modelo_dsa' (pipeline) cuida de todos os passos internos.
    """

    # 1. 'reviews' entra no pipeline
    # 2. TF-IDF é aplicado internamente
    # 3. StandardScaler é aplicado internamente
    # 4. LogisticRegression faz a previsão

    previsoes = modelo_deploy.predict(review)
    
    #mapeia resultado numero de volta para texto
    sentimentos = ['Negativo' if p == 0 else 'Positivo' for p in previsoes]

      # Exibe os resultados
    for review, sentimento in zip(review, sentimentos):
        print(f"\nReview: '{review}'\nSentimento Previsto: {sentimento}\n---")

# Executar a função de deploy com os novos dados
print("\n--- Iniciando Classificação de Novos Reviews (Deploy com Pipeline Completo) ---\n")
prever_sentimento(novos_reviews)