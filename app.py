from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

#cargar modelo
with open("model/vec_model.pkl", "rb") as file:
    vec_model = pickle.load(file)

data = np.load("model/similarity_matrix.npz")
similarity = data["arr_0"]
df = pd.read_csv("model/movies.csv")

 #Esta función es una adaptación de la que usé en el modelo
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