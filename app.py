from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation.hippopotamus import run_hippo_algorithm

app = Flask(__name__)
CORS(app)

@app.route("/api/simulate", methods=["POST"])
def simulate():
    data = request.get_json()
    algorithm = data.get("algorithm", "hippo")
    pop_size = int(data.get("pop_size", 30))
    max_iter = int(data.get("max_iter", 100))
    G = float(data.get("irradiance", 800))
    T = float(data.get("temperature", 25))

    if algorithm == "hippo":
        result = run_hippo_algorithm(pop_size, max_iter, G, T)
        return jsonify(result)
    else:
        return jsonify({"error": "Unsupported algorithm"}), 400

#Gunicorn - handles the server