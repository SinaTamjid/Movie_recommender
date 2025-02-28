import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity

#///////////////Preprocessing
# Load the datasets
users = pd.read_csv('users.csv')
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')

#///////////////Data Manipulation

#merging the datasets together with left join
merged_df=pd.merge(ratings,movies,on='movieId')
merged_df=pd.merge(merged_df,tags,on=['userId','movieId'],how='left')
# merged_df=merged_df.drop(columns=['tag'])

#random score for synthetic data
merged_df['sentiment'] = np.random.uniform(-1, 1, merged_df.shape[0])

#///////////////recommending system
#importing vader model
analyser=SentimentIntensityAnalyzer()

#using our vectorizer
vec=TfidfVectorizer(stop_words='english')
vec_matrix=vec.fit_transform(merged_df['title'] + " "+ merged_df['tag'].fillna(''))

#calculate cosine similarity
cosine_sim=cosine_similarity(vec_matrix,vec_matrix)

def recommendation(movie_title,cosine_sim=cosine_sim):
               idx= merged_df[merged_df['title']==movie_title].index[0]
               sim_score=list(enumerate(cosine_sim[idx]))
               sim_score=sorted(sim_score,key=lambda x: x[1],reverse=True)

               # Get the indices of the top 30 most similar movies (excluding itself)
               sim_score = [score for score in sim_score if score[0] != idx]
               sim_score = sim_score[:30]
               movie_indices=[i[0] for i in sim_score]
               # Return the unique titles of the top 30 recommended movies
               return merged_df['title'].iloc[movie_indices].unique().tolist()


recomend=recommendation('Movie 1')
print(recomend)



