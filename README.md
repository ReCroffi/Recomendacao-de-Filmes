# 🎬 Sistema de Recomendação de Filmes — NLP & Machine Learning

Sistema interativo que recomenda filmes por **similaridade de conteúdo** (sinopses),
usando NLP. Construído de ponta a ponta: coleta via API → limpeza → modelo → app web.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🚀 Funcionalidades
- **Motor de recomendação:** `TfidfVectorizer` + **similaridade de cosseno** para achar os 5 filmes mais próximos da sinopse escolhida.
- **EDA:** relação entre volume de lançamentos e notas médias ao longo dos anos.
- **Interface web:** app em Streamlit.
- **Pôsteres e notas em tempo real:** integração com a API do TMDB.

## 🧩 Decisões técnicas
- **Parsing de strings:** `ast` para converter listas em texto (ex.: gêneros) em objetos Python.
- **Normalização:** `explode` do Pandas para análise por gênero.
- **Stopwords PT-BR:** NLTK para limpar o texto antes do TF-IDF.
- **Segurança:** chave da API em `.env` (fora do versionamento).

## 🛠️ Stack
Python · Pandas · NumPy · scikit-learn (TF-IDF) · NLTK · Streamlit · TMDB API

## 📦 Dados
Filmes coletados da **API do TMDB** (`01_extração_api.py`) e salvos em
`filmes_populares.csv` / `filmes_populares_clean.csv`.

## ▶️ Como rodar
```bash
git clone git@github.com:ReCroffi/Recomendacao-de-Filmes.git
cd Recomendacao-de-Filmes
pip install -r requirements.txt
echo "TMDB_API_KEY=sua_chave_aqui" > .env
streamlit run app.py
```

## 🗂️ Estrutura
```
Recomendacao-de-Filmes/
├── 01_extração_api.py   # coleta dos dados (TMDB)
├── 02_EDA.py            # análise exploratória
├── 03_modelo.py         # TF-IDF + similaridade de cosseno
├── app.py               # interface Streamlit
└── requirements.txt
```

## ⚠️ Limitações e próximos passos
- Recomendação **só por conteúdo** (sinopse) — não usa avaliações de usuários,
  então sofre de "cold start" inverso (não aprende preferências individuais).
- Próximo: **filtragem colaborativa** para um sistema híbrido (conteúdo + comportamento).

---
Desenvolvido por **Renan Croffi** ([LinkedIn](https://www.linkedin.com/in/renancroffi/))
e **Raquel Duarte** ([LinkedIn](https://www.linkedin.com/in/raquel-duarte-1a2747397/)).
