import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

df = pd.read_csv('FinalMovies.csv')
# print(df.head())

df.set_index('title',inplace=True)
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
indices = pd.Series(df.index)

# function that takes in movie title as input and return top 10 recommended movies

def recommendations(title, cosine_sim = cosine_sim):
    recommended_movies = []
    
    # getting the index of the movie that matches the title
    idx = indices[indices == title].index[0]
    print('IDX', idx)
    
    # creating a Series with the similarity scores in descending order 
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    print('Score Series: ', score_series)
    
    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)
    print(top_10_indexes)
    
    # populating the list with the titles of the best 10  matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])
    
    return recommended_movies

print(recommendations('The Dark Knight'))

st.write("""
         # ~~~~~~Movie Recommender~~~~~~""")

# Choose Movie
movie = st.selectbox('Choose a Movie',(indices))
img_link = df.loc[movie,'image']
image_url = "{}".format(img_link)
st.image(image_url) 


# Recommendation Generator Button
if st.button('Hit Me So I Recommend You Movies!'):
    st.success("Movies Recommended")
    # Page Breaker
    st.write("""
            # --------Recommended Movies----------- """)

    # Get Recommendations
    recommended_movies = recommendations(movie)

    # Recommended Movie
    count = 1
    for i in recommended_movies:
        st.write(f"""
                ### Movie {count}: """)
        
        # image
        img_link = df.loc[i,'image']
        image_url = "{}".format(img_link)
        st.image(image_url)    
        
        # Movie
        st.write(f"""
                ###### Title:  """)
        rating = df.loc[i,'rating']
        st.write(i+' (Rating-{})'.format(rating))

        count += 1