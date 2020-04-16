######################################################################
######################################################################
######################################################################
#Part-2
######################################################################
######################################################################
######################################################################

users_df = pd.DataFrame(ratings_data['userId'].unique(), columns=['userId'])
users_df.head()

#create movies_df
movies_df = movies_data.drop('genres', axis = 1)
#calculate mean of ratings for each movies
agg_rating_avg = ratings_data.groupby(['movieId']).agg({'rating': np.mean}).reset_index()
agg_rating_avg.columns = ['movieId', 'rating_mean']
#merge
movies_df = movies_df.merge(agg_rating_avg, left_on='movieId', right_on='movieId', how='left')
movies_df.head()

genres = ["Action","Adventure","Animation","Children","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western","(no genres listed)"]
genres_df = pd.DataFrame(genres, columns=['genres'])
genres_df.head()

users_movies_df = ratings_data.drop('timestamp', axis = 1)
users_movies_df.head()

movies_genres_df = movies_data.drop('title', axis = 1)

#define a function to split genres field
def get_movie_genres(movieId):
    movie = movies_genres_df[movies_genres_df['movieId']==movieId]
    genres = movie['genres'].tolist()
    df = pd.DataFrame([b for a in [i.split('|') for i in genres] for b in a], columns=['genres'])
    df.insert(loc=0, column='movieId', value=movieId)
    return df

#create empty df
movies_genres=pd.DataFrame(columns=['movieId','genres'])
for x in movies_genres_df['movieId'].tolist():
    movies_genres=movies_genres.append(get_movie_genres(x))
print(movies_genres.head())

#join to movies data to get genre information
user_genres_df = ratings_data.merge(movies_data, left_on='movieId', right_on='movieId', how='left')
#drop columns that will not be used
user_genres_df.drop(['movieId','rating','timestamp','title'], axis = 1, inplace=True)
print(user_genres_df.head())

def get_favorite_genre(userId):
    user = user_genres_df[user_genres_df['userId']==userId]
    genres = user['genres'].tolist()
    movie_list = [b for a in [i.split('|') for i in genres] for b in a]
    counter = Counter(movie_list)
    return counter.most_common(1)[0][0]

#create empty df
users_genres = pd.DataFrame(columns=['userId','genre'])
for x in users_df['userId'].tolist():
    users_genres = users_genres.append(pd.DataFrame([[x,get_favorite_genre(x)]], columns=['userId','genre']))
print(users_genres.head())

users_df.to_csv('users.csv', sep='|', header=True, index=False)
movies_df.to_csv('movies.csv', sep='|', header=True, index=False)
genres_df.to_csv('genres.csv', sep='|', header=True, index=False)
users_movies_df.to_csv('users_movies.csv', sep='|', header=True, index=False)
movies_genres.to_csv('movies_genres.csv', sep='|', header=True, index=False)
users_genres.to_csv('users_genres.csv', sep='|', header=True, index=False)
movies_similarity.to_csv('movies_similarity.csv', sep='|', header=True, index=False)
