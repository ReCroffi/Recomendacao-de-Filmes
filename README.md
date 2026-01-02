# 🎬 Movie Recommendation System - NLP & Machine Learning

Este projeto é um sistema de recomendação de filmes interativo que utiliza técnicas de Processamento de Linguagem Natural (NLP) para sugerir títulos com base na similaridade de conteúdo (sinopses). O sistema foi construído de ponta a ponta: desde a coleta e limpeza dos dados até o deploy de uma interface web.

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🚀 Funcionalidades
- **Motor de Recomendação:** Utiliza `TfidfVectorizer` e `Similaridade de Cosseno` para encontrar filmes semanticamente próximos.
- **Análise de Dados (EDA):** Gráficos avançados que mostram a relação entre volume de lançamentos e notas médias ao longo dos anos.
- **Interface Web:** App interativo construído com Streamlit para uma experiência de usuário fluida.
- **Posters e Notas em Tempo Real:** Integração com a API do TMDB para exibir posters e avaliações atualizadas diretamente na interface.

## 📊 Insights e Desafios Técnicos
Durante o desenvolvimento, foram aplicadas soluções para desafios reais de Engenharia de Dados:
- **Tratamento de Strings Complexas:** Uso da biblioteca `ast` para converter representações textuais de listas em objetos Python reais.
- **Manipulação de Dados:** Uso do método `explode` do Pandas para normalizar gêneros e permitir análises estatísticas por categoria.
- **Segurança de Dados:** Implementação de variáveis de ambiente (`.env`) para proteção de chaves de API sensíveis.

## 🛠️ Tecnologias Utilizadas
- **Linguagem:** Python 3.x
- **Manipulação de Dados:** Pandas, NumPy
- **Visualização:** Seaborn, Matplotlib
- **NLP:** Scikit-Learn (TF-IDF), NLTK (Stopwords em Português)
- **Interface e API:** Streamlit, Requests

## 🔧 Como rodar o projeto localmente

1. **Clone o repositório:**
```bash
git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
```

2. **Instale as dependências:**

```bash
pip install -r requirements.txt
```
3. **Configure sua API Key: Crie um arquivo .env na raiz do projeto:**


```bash
TMDB_API_KEY=sua_chave_aqui
``` 
    
4. **Execute o App:**

```bash

streamlit run app.py
```
📈 **Resultados do Modelo**

O modelo consegue identificar padrões em sinopses de gêneros distintos, oferecendo recomendações precisas através de álgebra linear aplicada a texto. Ao selecionar um filme, o sistema calcula o peso das palavras-chave (TF-IDF) e sugere os 5 títulos com maior proximidade vetorial.

Desenvolvido por Renan Croffi - www.linkedin.com/in/renancroffi
e Raquel Duarte - https://www.linkedin.com/in/raquel-duarte-1a2747397/