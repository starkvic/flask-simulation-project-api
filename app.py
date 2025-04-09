from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation.algorithms.hippopotamus import run_hippo_algorithm
from simulation.algorithms.tlbo import run_tlbo_algorithm
import time
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

    start_time = time.time()
    
    if algorithm == "hippo":
        result = run_hippo_algorithm(pop_size, max_iter, G, T)
    elif algorithm == "tlbo":
        result = run_tlbo_algorithm(pop_size, max_iter, G, T) 
    else:
        return jsonify({"error": "Unsupported algorithm"}), 400
    
    result["runtime"] = round(time.time() - start_time, 4)
    return jsonify(result)


@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/test")
def test():
    return jsonify({"message": "test route is working"})

#Gunicorn - handles the server
#reverted server changes
