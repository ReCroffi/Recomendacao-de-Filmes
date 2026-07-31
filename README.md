# 🎬 Sistema de Recomendação de Filmes — NLP & Machine Learning

Recomendador de filmes por **similaridade de conteúdo**: dada uma sinopse, o sistema encontra os
5 filmes mais próximos usando TF-IDF e similaridade de cosseno. Construído de ponta a ponta —
coleta via API → limpeza → modelagem → aplicação web.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f5b?style=for-the-badge)

---

## Pipeline

```
TMDB API  →  01_extração_api.py  →  filmes_populares.csv       (1.000 registros brutos)
                    ↓ limpeza
                                     filmes_populares_clean.csv  (980 registros)
                    ↓
          02_EDA.py       análise exploratória
          03_modelo.py    TF-IDF + similaridade de cosseno
                    ↓
             app.py       aplicação Streamlit
```

Cada etapa é um script independente que lê e grava em disco — o modelo pode ser reconstruído sem
refazer a coleta.

---

## Dados

Coletados do endpoint `/movie/top_rated` da **API do TMDB**, em português (`language=pt-BR`),
percorrendo 50 páginas.

| | |
|---|---|
| Registros brutos | 1.000 |
| Após limpeza | 980 |
| Descartados | 20 (sinopse vazia ou data inválida) |
| Período | 1902 – 2025 |
| Nota média | 7,90 (faixa 7,6 – 8,7) |
| Tamanho médio da sinopse | 58,8 palavras |

**Limpeza aplicada** (`01_extração_api.py`): remoção de sinopses nulas ou vazias — são a única
entrada do modelo —, conversão de `release_date` para datetime com descarte de datas inválidas,
e derivação das colunas `year` e `month`.

> ⚠️ **Sobre o nome dos arquivos:** os CSVs se chamam `filmes_populares`, mas a origem é o endpoint
> `top_rated`, não `popular`. São os filmes mais **bem avaliados**, não os mais populares. Isso
> restringe as notas à faixa 7,6–8,7 e é o principal viés do dataset — ver *Achados da EDA*.

---

## Motor de recomendação

Filtragem **baseada em conteúdo**: cada sinopse vira um vetor TF-IDF, e a recomendação são os
vizinhos mais próximos por cosseno.

```python
tfidf = TfidfVectorizer(stop_words=stopwords.words('portuguese'))
tfidf_matrix = tfidf.fit_transform(df['overview'])   # (980, 10290)
cosine_sim = cosine_similarity(tfidf_matrix)
```

| | |
|---|---|
| Matriz TF-IDF | 980 × 10.290 |
| Vocabulário | 10.290 termos |
| Densidade | 0,31% (matriz esparsa) |
| Stopwords PT-BR | 207 termos (NLTK) |
| Matriz de similaridade | 980 × 980 · ~7,7 MB em memória |

No Streamlit a matriz é construída uma única vez e mantida em cache com `@st.cache_resource`, o que
evita recalcular a similaridade a cada interação.

---

## Achados da EDA

**Sazonalidade de lançamentos é real e forte.** Dezembro concentra 138 lançamentos e setembro 122,
contra 48 em janeiro — quase 3× de diferença entre o pico e o vale. Há também uma elevação em junho
(94), consistente com a temporada de férias do hemisfério norte, mas o pico verdadeiro é o de fim de
ano.

**Nota não acompanha o calendário.** A amplitude da nota média entre o melhor e o pior mês é de
**0,06 ponto** — ruído. Quando um filme é lançado não diz nada sobre como ele será avaliado.

**Gênero também quase não separa nota.** Da Música (7,95) ao Terror (7,80), a amplitude é de
**0,14 ponto** entre 19 gêneros.

**A explicação para os dois achados anteriores está no dataset, não no cinema.** Como a coleta usa
`top_rated`, todos os 980 filmes já passaram por um filtro de qualidade — a variável resposta está
comprimida numa faixa de 1,1 ponto. Nenhuma variável vai explicar variação que foi removida na
amostragem. Para investigar de verdade o que dirige nota seria preciso coletar também
`/movie/popular` ou `/discover`, que incluem filmes mal avaliados.

**Volume cresce, qualidade não segue.** 39% do catálogo é de 2013 em diante, com pico em 2019
(50 filmes). 2022 foi um dos anos de maior volume e ficou com nota média de 7,83, abaixo da média
geral.

---

## Avaliação do modelo

Não existe ground truth de "filme parecido" neste dataset, então não há acurácia a reportar. O que
dá para medir é **a distribuição das similaridades** — e ela expõe o limite da abordagem:

| Métrica (similaridade do melhor match de cada filme) | Valor |
|---|---|
| Média | 0,129 |
| Mediana | 0,114 |
| Filmes cujo melhor match fica abaixo de 0,15 | **766 de 980 (78%)** |

Em cosseno, 0,13 é um sinal fraco. A causa é estrutural: sinopses do TMDB têm ~59 palavras, e duas
sinopses curtas raramente compartilham vocabulário suficiente para gerar proximidade alta, mesmo
quando os filmes são de fato parecidos.

Na prática o resultado é **irregular** — funciona quando há vocabulário narrativo em comum e degrada
para ruído quando não há:

```
>>> O Poderoso Chefão
    0.081  Parasita
    0.079  Ainda Temos o Amanhã
    0.074  O Poderoso Chefão: Parte II     ← acerto real
    0.074  Com Amor, Simon
    0.071  O Bebê de Rosemary

>>> Interestelar
    0.116  Contra-Ataque
    0.097  A Lenda de Hei                  ← sem relação temática
    0.088  Próxima Parada: Lar Doce Lar
```

O caso do Poderoso Chefão traz a sequência correta em terceiro lugar, mas cercada de filmes sem
parentesco. Interestelar é ruído puro. **A conclusão honesta é que TF-IDF sobre sinopse curta não é
suficiente** — o que motiva os próximos passos abaixo.

---

## Limitações e bugs conhecidos

**Do modelo**

- **Só conteúdo, sem comportamento.** Não usa avaliações nem histórico de usuário, então não
  personaliza: dois usuários que escolhem o mesmo filme recebem exatamente a mesma lista.
- **Sinal fraco** — ver a seção anterior. 78% das recomendações partem de similaridade < 0,15.
- **Não escala como está.** A matriz de similaridade é O(n²): 980 filmes ocupam 7,7 MB, mas 50 mil
  filmes passariam de 20 GB. Um catálogo maior exigiria busca aproximada por vizinhos (FAISS,
  Annoy) em vez da matriz completa.
- **Viés de amostragem** do endpoint `top_rated`, descrito em *Dados*.

**Bugs identificados no código atual**

- **Registros duplicados no dataset.** `Guerreiras do K-Pop` e `Marcados Pelo Sangue` aparecem duas
  vezes com o **mesmo `id`** do TMDB. Consequência direta: o par mais similar de todo o dataset é um
  filme com ele mesmo (cosseno = 1,000), e o app recomenda o próprio filme selecionado como primeira
  sugestão. Corrigir com `drop_duplicates(subset='id')` na limpeza.
- **Remakes colidem por título.** Sete títulos se repetem com `id` diferente — `12 Homens e uma
  Sentença` (1957 e 1997), `Como Treinar o Seu Dragão` (2010 e 2025), `Os Suspeitos` (1995 e 2013),
  entre outros. Como `app.py` resolve a seleção por `df[df['title'] == selected].index[0]`, escolher
  o remake sempre devolve as recomendações do original. A seleção deveria ser por `id`, com o ano
  exibido no seletor.
- **Fallback de imagem quebrado.** `get_poster()` devolve uma URL de `via.placeholder.com` em caso de
  erro, e esse serviço está fora do ar — a falha de rede vira uma imagem quebrada. Melhor tratar com
  um placeholder local ou `st.empty()`.
- **`except` silencioso.** O `try/except` de `get_poster()` captura qualquer exceção sem log, o que
  esconde erro de chave de API e erro de rede sob o mesmo comportamento.
- **`nltk.download('stopwords')` roda a cada start** do app, incluindo uma ida à rede em toda
  inicialização. Deveria ser condicional a um `LookupError`.

---

## Próximos passos

1. **Corrigir a deduplicação por `id`** e passar a resolver a seleção por `id` em vez de título — as
  duas correções de maior impacto e menor custo.
2. **Enriquecer o vetor de conteúdo** concatenando gênero, diretor e elenco à sinopse antes do
  TF-IDF. É o caminho mais direto para tirar a similaridade média da faixa de 0,13, já que ataca
  justamente a escassez de vocabulário.
3. **Trocar TF-IDF por embeddings de sentença** (ex.: `sentence-transformers`), que capturam
  similaridade semântica em vez de sobreposição literal de palavras — dois filmes podem ser
  parecidos sem repetir nenhuma palavra.
4. **Filtragem colaborativa** sobre avaliações de usuários, para um sistema híbrido conteúdo +
  comportamento.
5. **Avaliação com ground truth**: usar as coleções e os `similar` do próprio TMDB como referência
  para medir precision@5 e ter um número comparável entre versões.

---

## Como rodar

```bash
git clone https://github.com/ReCroffi/Recomendacao-de-Filmes.git
cd Recomendacao-de-Filmes

python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

echo "TMDB_API_KEY=sua_chave_aqui" > .env         # chave gratuita em themoviedb.org/settings/api
streamlit run app.py
```

Os CSVs já estão versionados, então o app roda sem refazer a coleta. A chave do TMDB é necessária
apenas para os pôsteres em tempo real — para regerar os dados do zero, rode `01_extração_api.py`.

---

## Estrutura

```
Recomendacao-de-Filmes/
├── 01_extração_api.py           # coleta no TMDB + limpeza → CSVs
├── 02_EDA.py                    # análise exploratória (sazonalidade, gênero, ano)
├── 03_modelo.py                 # TF-IDF + similaridade de cosseno + função de recomendação
├── app.py                       # interface Streamlit com pôsteres e notas
├── filmes_populares.csv         # dados brutos (1.000)
├── filmes_populares_clean.csv   # dados tratados (980)
└── requirements.txt
```

**Stack:** Python · Pandas · NumPy · scikit-learn · NLTK · Streamlit · Matplotlib · Seaborn · TMDB API

**Decisões técnicas:** `ast.literal_eval` para reconstruir as listas de gênero salvas como string no
CSV · `explode` do Pandas para normalizar a análise por gênero · stopwords PT-BR do NLTK antes da
vetorização · chave de API em `.env`, fora do versionamento.

---

Desenvolvido por **Renan Croffi** ([LinkedIn](https://www.linkedin.com/in/renancroffi/)) e
**Raquel Duarte** ([LinkedIn](https://www.linkedin.com/in/raquel-duarte-1a2747397/)).
