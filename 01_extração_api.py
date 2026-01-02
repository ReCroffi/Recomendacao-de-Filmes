# %% 
import os
import pandas as pd 
import requests
from dotenv import load_dotenv
from matplotlib import pyplot as plt 
import seaborn as sns

# %%
def get_data():
    load_dotenv()
    api_key = os.getenv('TMDB_API_KEY')
    filmes = []

    for page in range(1, 51):
        url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={api_key}&language=pt-BR&page={page}"
        resp = requests.get(url=url)
    
        if resp.status_code == 200:
            data = resp.json()
            for filme in data.get('results', []):
                filmes.append({
                    'id': filme.get('id'),
                    'title': filme.get('title'),
                    'release_date': filme.get('release_date'),
                    'vote_average': filme.get('vote_average'),
                    'overview': filme.get('overview'), # Sinopse para o NLP
                    'genre_ids': filme.get('genre_ids')
                })
    return filmes                  
# %%
filmes = get_data()
#%% 
df = pd.DataFrame(filmes)
df.head()
df.describe()
# %%
df.to_csv('filmes_populares.csv', index=False)
# %%
#como faremos recomendação baseada na sinopse vou garantir que não tenha N/A

df = df.dropna(subset=['overview'])
df = df[df['overview'] != ""]
# %%
# %%
df['release_date'] = pd.to_datetime(df['release_date'], errors= 'coerce')
df=df.dropna(subset=['release_date'])
# %%
df['year']= df['release_date'].dt.year
# %%
df['month']= df['release_date'].dt.month
# %%
df
# %%
df.to_csv('filmes_populares_clean.csv', index=False)
# %%
