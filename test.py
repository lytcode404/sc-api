import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your data
data = pd.read_csv('Constitution of India.csv')
df = pd.DataFrame(data)

# Input text (the case you're dealing with)
input_text = "I need information about the formation of new states."

# Text preprocessing and Vectorization
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the descriptions and input text
# Assuming 'description' column has article content
tfidf_matrix = vectorizer.fit_transform(df['description'])
input_vec = vectorizer.transform([input_text])

# Calculate cosine similarity between input text and descriptions
cosine_sim = cosine_similarity(input_vec, tfidf_matrix)

# Get the indices of the most similar articles sorted by similarity score
top_n = 5  # Number of top matches you want to return
# Sort and get top N indices
top_indices = cosine_sim.argsort()[0][-top_n:][::-1]

# Display the most relevant articles
for idx in top_indices:
    article_number = df.iloc[idx]['article']
    title = df.iloc[idx]['title']
    similarity_score = cosine_sim[0][idx]
    print(
        f"Article Number: {article_number}, Title: {title}, Similarity Score: {similarity_score:.4f}")
