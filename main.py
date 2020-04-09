# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 19:12:05 2020

@author: kanis
"""

import pandas as pd
import numpy as np
import datetime
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

genome_scores_data = pd.read_csv("genome-scores.csv") 
movies_data = pd.read_csv("movies.csv") 
ratings_data = pd.read_csv("ratings.csv")

genome_scores_data.head()

movies_data.head()

ratings_data.head()

scores_pivot = genome_scores_data.pivot_table(index = ["movieId"],columns = ["tagId"],values = "relevance").reset_index()
scores_pivot.head()

#join
mov_tag_df = movies_data.merge(scores_pivot, left_on="movieId", right_on="movieId", how="left")
mov_tag_df = mov_tag_df.fillna(0) 
mov_tag_df = mov_tag_df.drop(['title','genres'], axis = 1)
mov_tag_df.head()

mov_genres_df = pd.read_csv("movies.csv") 

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
mov_genres_df.head()

def set_year(title):
    year = title.strip()[-5:-1]
    if unicode(year, 'utf-8').isnumeric() == True: return int(year)
    else: return 1800
    

#join
movies = movies_data.merge(scores_pivot, left_on="movieId", right_on="movieId", how="left")
movies = movies.fillna(0) 
movies = movies.drop(['title','genres'], axis = 1)
movies.head()


#add year field
movies['year'] = movies.apply(lambda x: set_year(x['title']), axis=1)
movies = movies_data.drop('genres', axis = 1)
movies.head()