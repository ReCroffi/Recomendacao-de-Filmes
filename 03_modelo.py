#%% 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords

#%% 
nltk.download('stopwords')
palavras_irrelevantes = nltk.corpus.stopwords.words('portuguese')

df = pd.read_csv('filmes_populares_clean.csv')

tfidf = TfidfVectorizer(stop_words=palavras_irrelevantes)

#garantir que não tenha null
df['overview'] = df['overview'].fillna('')

#matriz TD-IDF
tfidf_matrix = tfidf.fit_transform(df['overview'])

#formato da matriz
print(tfidf_matrix.shape)
# %%
# Calcular a matriz de similaridade (isso pode demorar se o dataset for gigante)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# %%

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# %%
def recomendacao(titulo, cosine_sim =cosine_sim):
    if titulo not in indices:
        return 'Filme não encontrado'
    idx = indices[titulo]
    
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices]

# %%

# %%
