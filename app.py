from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

questions = []
answers = []

# Load CSV
with open("knowledge111.csv", newline='', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    for row in reader:
        questions.append(row["question"])
        answers.append(row["answer"])

# TF-IDF Model
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_bot_response(user_message):
    user_vector = vectorizer.transform([user_message])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score > 0.2:  # similarity threshold
        return answers[best_match_index]
    else:
        return "Sorry 😔 mujhe samajh nahi aaya."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    reply = get_bot_response(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
