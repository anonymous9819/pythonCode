import pandas as pd
import numpy as np
import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

genome_scores_data = pd.read_csv('genome-scores.csv') 
movies_data = pd.read_csv('movies.csv') 
ratings_data = pd.read_csv('ratings.csv')

genome_scores_data.head()
movies_data.head()
ratings_data.head()

scores_pivot = genome_scores_data.pivot_table(index = ["movieId"],columns = ["tagId"],values = "relevance").reset_index()
scores_pivot.head()

#join
mov_tag_df = movies_data.merge(scores_pivot, left_on='movieId', right_on='movieId', how='left')
mov_tag_df = mov_tag_df.fillna(0) 
mov_tag_df = mov_tag_df.drop(['title','genres'], axis = 1)
mov_tag_df.head()

#mov_genres_df is created with using “movies.csv”. We split genres field for each movies and then we create columns for each genres. We define a function to split genres column and check it if it exists or not
mov_genres_df= pd.read_csv('movies.csv') 

def set_genres(genres,col):
    if genres in col.split('|'): return 1
    else: return 0

mov_genres_df["Action"] = mov_genres_df.apply(lambda x: set_genres("Action",x['genres']), axis=1)
mov_genres_df["Adventure"] = mov_genres_df.apply(lambda x: set_genres("Adventure",x['genres']), axis=1)
mov_genres_df["Animation"] = mov_genres_df.apply(lambda x: set_genres("Animation",x['genres']), axis=1)
mov_genres_df["Children"] = mov_genres_df.apply(lambda x: set_genres("Children",x['genres']), axis=1)
mov_genres_df["Comedy"] = mov_genres_df.apply(lambda x: set_genres("Comedy",x['genres']), axis=1)
mov_genres_df["Crime"] = mov_genres_df.apply(lambda x: set_genres("Crime",x['genres']), axis=1)
mov_genres_df["Documentary"] = mov_genres_df.apply(lambda x: set_genres("Documentary",x['genres']), axis=1)
mov_genres_df["Drama"] = mov_genres_df.apply(lambda x: set_genres("Drama",x['genres']), axis=1)
mov_genres_df["Fantasy"] = mov_genres_df.apply(lambda x: set_genres("Fantasy",x['genres']), axis=1)
mov_genres_df["Film-Noir"] = mov_genres_df.apply(lambda x: set_genres("Film-Noir",x['genres']), axis=1)
mov_genres_df["Horror"] = mov_genres_df.apply(lambda x: set_genres("Horror",x['genres']), axis=1)
mov_genres_df["Musical"] = mov_genres_df.apply(lambda x: set_genres("Musical",x['genres']), axis=1)
mov_genres_df["Mystery"] = mov_genres_df.apply(lambda x: set_genres("Mystery",x['genres']), axis=1)
mov_genres_df["Romance"] = mov_genres_df.apply(lambda x: set_genres("Romance",x['genres']), axis=1)
mov_genres_df["Sci-Fi"] = mov_genres_df.apply(lambda x: set_genres("Sci-Fi",x['genres']), axis=1)
mov_genres_df["Thriller"] = mov_genres_df.apply(lambda x: set_genres("Thriller",x['genres']), axis=1)
mov_genres_df["War"] = mov_genres_df.apply(lambda x: set_genres("War",x['genres']), axis=1)
mov_genres_df["Western"] = mov_genres_df.apply(lambda x: set_genres("Western",x['genres']), axis=1)
mov_genres_df["(no genres listed)"] = mov_genres_df.apply(lambda x: set_genres("(no genres listed)",x['genres']), axis=1)

mov_genres_df.drop(['title','genres'], axis = 1, inplace=True)

def set_year(title):
    year = title.strip()[-5:-1]
    if year.isnumeric() == True: return int(year)
    else: return 1800

#add year field
movies=pd.read_csv('movies.csv')
movies['year'] = movies.apply(lambda x: set_year(x['title']), axis=1)
movies = movies.drop('genres', axis = 1)

#define function to group years
def set_year_group(year):
    if (year < 1900): return 0
    elif (1900 <= year <= 1975): return 1
    elif (1976 <= year <= 1995): return 2
    elif (1996 <= year <= 2003): return 3
    elif (2004 <= year <= 2009): return 4
    elif (2010 <= year): return 5
    else: return 0
movies['year_group'] = movies.apply(lambda x: set_year_group(x['year']), axis=1)
#no need title and year fields
movies.drop(['title','year'], axis = 1, inplace=True)

agg_movies_rat = ratings_data.groupby(['movieId']).agg({'rating': [np.size, np.mean]}).reset_index()
agg_movies_rat.columns = ['movieId','rating_counts', 'rating_mean']
print(agg_movies_rat.head())

#define function to group rating counts
def set_rating_group(rating_counts):
    if (rating_counts <= 1): return 0
    elif (2 <= rating_counts <= 10): return 1
    elif (11 <= rating_counts <= 100): return 2
    elif (101 <= rating_counts <= 1000): return 3
    elif (1001 <= rating_counts <= 5000): return 4
    elif (5001 <= rating_counts): return 5
    else: return 0
agg_movies_rat['rating_group'] = agg_movies_rat.apply(lambda x: set_rating_group(x['rating_counts']), axis=1)
#no need rating_counts field
agg_movies_rat.drop('rating_counts', axis = 1, inplace=True)
mov_rating_df = movies.merge(agg_movies_rat, left_on='movieId', right_on='movieId', how='left')
mov_rating_df = mov_rating_df.fillna(0)
mov_rating_df.head()

mov_tag_df = mov_tag_df.set_index('movieId')
mov_genres_df = mov_genres_df.set_index('movieId')
mov_rating_df = mov_rating_df.set_index('movieId')

#cosine similarity for mov_tag_df
cos_tag = cosine_similarity(mov_tag_df.values)*0.5
#cosine similarity for mov_genres_df
cos_genres = cosine_similarity(mov_genres_df.values)*0.25
#cosine similarity for mov_rating_df
cos_rating = cosine_similarity(mov_rating_df.values)*0.25
#mix
cos = cos_tag+cos_genres+cos_rating

cols = mov_tag_df.index.values
inx = mov_tag_df.index
movies_sim = pd.DataFrame(cos, columns=cols, index=inx)
print(movies_sim.head())

def get_similar(movieId):
    df = movies_sim.loc[movies_sim.index == movieId].reset_index().melt(id_vars='movieId', var_name='sim_moveId', value_name='relevance').sort_values('relevance', axis=0, ascending=False)[1:6]
    return df
#create empty df
movies_similarity = pd.DataFrame(columns=['movieId','sim_moveId','relevance'])

for x in movies_sim.index.tolist():
    movies_similarity = movies_similarity.append(get_similar(x))
print(movies_similarity.head())

def movie_recommender(movieId):
    df = movies_sim.loc[movies_sim.index == movieId].reset_index().melt(id_vars='movieId', var_name='sim_moveId', value_name='relevance').sort_values('relevance', axis=0, ascending=False)[1:6]
    df['sim_moveId'] = df['sim_moveId'].astype(int)
    sim_df = movies_data.merge(df, left_on='movieId', right_on='sim_moveId', how='inner').sort_values('relevance', axis=0, ascending=False).loc[: , ['movieId_y','title','genres']].rename(columns={ 'movieId_y': "movieId" })
    return sim_df

print(movie_recommender(1))
