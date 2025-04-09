from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Cargar datos
df = pd.read_csv("movies.csv")

# Crear el modelo de vectorización y calcular los vectores
vec_model = CountVectorizer(stop_words="english")
# Suponiendo que el campo de interés es 'tags' (ajusta según tus datos)
vectors = vec_model.fit_transform(df['tags']).toarray()

# Calcular la matriz de similitud de coseno sobre los vectores
similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in df["title"].values:
        return ["Movie not found"]
    
    movie_index = df[df["title"] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [df.iloc[i[0]].title for i in movie_list]

# Página principal
@app.route("/")
def index():
    return render_template("index.html")

# Endpoint para obtener recomendaciones
@app.route("/recommend", methods=["POST"])
def get_recommendations():
    movie = request.form.get("movie")
    recommendations = recommend(movie)
    return jsonify({"recommendations": recommendations})

if __name__ == "__main__":
    app.run(debug=True)
