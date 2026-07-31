
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


#só baixa se ainda não estiver em cache, em vez de ir na rede a cada start
try:
    palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')
    palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')
#%%
#config da pagina
st.set_page_config(page_title="CineRecommend", layout="wide")

# %%
def get_poster(movie_id):
    """Devolve a URL do pôster no TMDB, ou None se não houver/falhar."""
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=pt-BR"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        path = resp.json().get('poster_path')
    except requests.RequestException as erro:
        #antes era um except mudo: falha de rede e chave inválida ficavam iguais
        st.warning(f"Falha ao buscar o pôster (id {movie_id}): {erro}")
        return None
    return f"https://image.tmdb.org/t/p/w500{path}" if path else None


def poster_indisponivel():
    """Placeholder desenhado localmente — o serviço externo antigo saiu do ar."""
    st.markdown(
        "<div style='aspect-ratio:2/3;display:flex;align-items:center;justify-content:center;"
        "background:#262730;border:1px solid #3d3d46;border-radius:8px;color:#8a8a95;"
        "font-size:.8rem;text-align:center;padding:8px'>pôster<br>indisponível</div>",
        unsafe_allow_html=True,
    )

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
#a seleção devolve a posição da linha, não o título: títulos se repetem entre
#remakes ("Os Suspeitos" 1995 e 2013) e buscar por título pegava sempre o primeiro
def rotulo(pos):
    linha = df.iloc[pos]
    ano = int(linha['year']) if pd.notna(linha['year']) else '?'
    return f"{linha['title']} ({ano})"

idx = st.selectbox("Escolha um filme:", range(len(df)), format_func=rotulo)


if st.button('Recomendar'):
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
            if poster_url:
                st.image(poster_url, width='stretch')
            else:
                poster_indisponivel()

            # 2. Exibe a nota com um emoji de estrela para ficar visual
            st.markdown(f"⭐ **{movie_rating:.1f}**/10")
            
            # 3. Exibe o título
            st.caption(f"**{movie_title}**")