from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load your data and prepare the TF-IDF vectorizer and matrix
data = pd.read_csv('Constitution of India.csv')
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['description'])


@app.route('/search', methods=['POST'])
def search_articles():
    # Get the input text from the POST request
    input_data = request.json
    input_text = input_data.get('input_text')

    # Transform the input text using the existing vectorizer
    input_vec = vectorizer.transform([input_text])

    # Calculate cosine similarity between input text and descriptions
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix)

    # Get the indices of the most similar articles sorted by similarity score
    top_n = 5  # Number of top matches you want to return
    top_indices = cosine_sim.argsort()[0][-top_n:][::-1]

    # Prepare the results as a list of dictionaries
    results = []
    for idx in top_indices:
        article_number = df.iloc[idx]['article']
        title = df.iloc[idx]['title']
        similarity_score = cosine_sim[0][idx]
        results.append({
            'article_number': article_number,
            'title': title,
            'similarity_score': round(similarity_score, 4)
        })

    # Return the results as JSON
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)
