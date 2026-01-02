
#%%
import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import nltk
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('TMDB_API_KEY')


nltk.download('stopwords')
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
#%%
#config da pagina
st.set_page_config(page_title="CineRecommend", layout="wide")

# %%
def get_poster(movie_id):
    api_key = API_KEY
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=pt-BR"
    try:
        data = requests.get(url).json()
        path = data.get('poster_path')
        if path:
            full_path = "https://image.tmdb.org/t/p/w500" + path
            print(f"Link gerado: {full_path}") # Isso aparece no seu terminal/prompt
            return full_path
    except:
        pass
    return "https://via.placeholder.com/500x750?text=Erro+na+URL"

#%%
@st.cache_resource
def build_model(df):
    tfidf = TfidfVectorizer(stop_words=palavras_irrelevantes)
    tfidf_matrix = tfidf.fit_transform(df['overview'].fillna(''))
    return cosine_similarity(tfidf_matrix)
# %%
st.title("🎬 Sistema de Recomendação de Filmes")
df = pd.read_csv("filmes_populares_clean.csv")
cosine_sim = build_model(df)
# %%
#
# %%
selected_movie = st.selectbox("Escolha um filme:", df['title'].values)



if st.button('Recomendar'):
    idx = df[df['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(cosine_sim[idx])), reverse=True, key=lambda x: x[1])[1:6]
    
    cols = st.columns(5)
    
    for i, dist in enumerate(distances):
        movie_row = df.iloc[dist[0]]
        movie_title = movie_row['title']
        movie_id = movie_row['id']
        # 1. Captura a nota média do DataFrame
        movie_rating = movie_row['vote_average'] 
        
        poster_url = get_poster(movie_id)
        
        with cols[i]:
            st.image(poster_url, width='stretch')
            
            # 2. Exibe a nota com um emoji de estrela para ficar visual
            st.markdown(f"⭐ **{movie_rating:.1f}**/10")
            
            # 3. Exibe o título
            st.caption(f"**{movie_title}**")