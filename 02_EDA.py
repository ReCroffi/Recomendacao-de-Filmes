# %% 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import ast

# %%
df = pd.read_csv('filmes_populares_clean.csv')
df['genre_ids'] = df['genre_ids'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df.head()



#%%
#Existe relação entre o mes de lançamento e o sucesso dos filmes?
plt.figure(figsize=(15, 5))

# Gráfico 1: Média de Notas
plt.subplot(1, 2, 1)
sns.barplot(data=df, x='month', y='vote_average')
plt.title('Nota Média por Mês')

# Gráfico 2: Quantidade de Filmes
plt.subplot(1, 2, 2)
sns.countplot(data=df, x='month')
plt.title('Total de Lançamentos por Mês')

plt.tight_layout()
plt.show()# %%

# %%
#Podemos ver uma sazionalidade no numero de lançamentos por mes, com um boom no verão do hemisferio norte (junho-julho)
# e subindo novamente no outono com um pico no inverno(dezembro)
# o pico do verão deve-se as ferias escolares
# e o do inverno pode ser porque as pessoas ficam mais em casa devido a neve, e também muitos
#filmes que remetem ao clima de fim de ano, como natal e ano novo
# %%
# O genero impacta a nota?

df_plot = df.explode('genre_ids')
generos_nomes = {
    28: 'Ação', 12: 'Aventura', 16: 'Animação', 35: 'Comédia', 80: 'Crime',
    99: 'Doc', 18: 'Drama', 10751: 'Família', 14: 'Fantasia', 36: 'História',
    27: 'Terror', 10402: 'Música', 9648: 'Mistério', 10749: 'Romance',
    878: 'Ficção', 10770: 'Cinema TV', 53: 'Suspense', 10752: 'Guerra', 37: 'Faroeste'
}
df_plot['genre_name'] = df_plot['genre_ids'].map(generos_nomes)
df_plot = df_plot.dropna(subset=['genre_name'])
#%%
plt.figure(figsize=(12, 6))
order = df_plot.groupby('genre_name')['vote_average'].mean().sort_values(ascending=False).index
sns.barplot(data=df_plot, x='genre_name', y='vote_average', palette='magma',order=order)
plt.xticks(rotation=45) 
plt.title('Média de Notas por Gênero')
plt.show()
# %%
#Não parece ter uma discrepancia muito grande entre as notas. 
#Esse dataset contem os filmes top rated, por isso não vemos muita variação

#%%
df_agrupado = df.groupby('year').agg({
    'vote_average': 'mean',
    'title': 'count'
}).reset_index().rename(columns={'title': 'contagem'})

# 1. Filtrar para anos recentes
df_recente = df_agrupado[df_agrupado['year'] >= 1990].sort_values('year')

fig, ax1 = plt.subplots(figsize=(14, 7))

# Estética de fundo
sns.set_style("white")

# Eixo 1: Barras (Volume)
sns.barplot(data=df_recente, x='year', y='contagem', ax=ax1, color='#AED6F1', alpha=0.7)
ax1.set_ylabel('Volume de Lançamentos', color='#2E86C1', fontsize=12, fontweight='bold')
ax1.set_xlabel('Ano de Lançamento', fontsize=11)
ax1.tick_params(axis='x', rotation=45)

# Eixo 2: Linha (Notas)
ax2 = ax1.twinx()

sns.lineplot(data=df_recente, x=range(len(df_recente)), y='vote_average', ax=ax2, 
             color='#C0392B', marker='o', markersize=6, linewidth=3, label='Nota Média')


ax2.set_ylim(df_recente['vote_average'].min() - 0.5, df_recente['vote_average'].max() + 0.5)
ax2.set_ylabel('Nota Média (TMDB)', color='#C0392B', fontsize=12, fontweight='bold')


sns.despine(ax=ax1, top=True)
sns.despine(ax=ax2, top=True, right=False)

plt.title('Evolução do Mercado de Filmes: Volume vs. Qualidade (1990-2025)', fontsize=15, pad=20)
plt.tight_layout()
plt.show()
# %%

# %%
#Vemos um crescimento nos lançamentos a patir do ano de 2013
#Mas as notas não acompanham, veja 2022 como exemplo, foi o segundo ano com mais lancamentos
#e vemos uma queda na média das notas, isso nos diz que volume nem sempre significa qualidade

# %%
