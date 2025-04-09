from flask import Flask, request, jsonify
from simulation.algorithms import (
    run_hippo_algorithm,
    run_tlbo_algorithm,
    run_genetic_algorithm,
    run_pso_algorithm,
    run_sa_algorithm,
    run_gwo_algorithm,
    run_hs_algorithm,
    run_csa_algorithm,
    run_lsa_algorithm,
    run_epo_algorithm,
    run_cs_algorithm,
    run_abo_algorithm,
    run_hem_algorithm,
    run_wso_algorithm,
    run_de_algorithm,
    run_abc_algorithm
)

app = Flask(__name__)

# Mapping algorithm names to their functions
ALGORITHM_MAP = {
    "hippo": run_hippo_algorithm,
    "tlbo": run_tlbo_algorithm,
    "ga": run_genetic_algorithm,
    "pso": run_pso_algorithm,
    "sa": run_sa_algorithm,
    "gwo": run_gwo_algorithm,
    "hs": run_hs_algorithm,
    "csa": run_csa_algorithm,
    "lsa": run_lsa_algorithm,
    "epo": run_epo_algorithm,
    "cs": run_cs_algorithm,
    "abo": run_abo_algorithm,
    "hem": run_hem_algorithm,
    "wso": run_wso_algorithm,
    "de": run_de_algorithm,
    "abc": run_abc_algorithm
}

@app.route("/run-algorithm", methods=["POST"])
def run_algorithm():
    data = request.json
    algorithm = data.get("algorithm")
    G = data.get("irradiance", 800)
    T = data.get("temperature", 25)
    pop_size = data.get("pop_size", 30)
    max_iter = data.get("max_iter", 100)

    if algorithm not in ALGORITHM_MAP:
        return jsonify({"error": "Invalid algorithm name."}), 400

    try:
        result = ALGORITHM_MAP[algorithm](pop_size, max_iter, G, T)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
