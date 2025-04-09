from flask import Flask, request, jsonify
from simulation.hippopotamus import run_hippo_algorithm
from simulation.tlbo import run_tlbo_algorithm
from simulation.genetic import run_genetic_algorithm
from simulation.pso import run_pso_algorithm
from simulation.sa import run_sa_algorithm
from simulation.gwo import run_gwo_algorithm
from simulation.hs import run_hs_algorithm
from simulation.csa import run_csa_algorithm
from simulation.lsa import run_lsa_algorithm
from simulation.epo import run_epo_algorithm
from simulation.cs import run_cs_algorithm
from simulation.abo import run_abo_algorithm
from simulation.hem import run_hem_algorithm
from simulation.wso import run_wso_algorithm
from simulation.de import run_de_algorithm
from simulation.abc import run_abc_algorithm

app = Flask(__name__)

@app.route("/run-algorithm", methods=["POST"])
def run_algorithm():
    data = request.json
    algorithm = data.get("algorithm")
    G = data.get("irradiance", 800)
    T = data.get("temperature", 25)
    pop_size = data.get("pop_size", 30)
    max_iter = data.get("max_iter", 100)

    try:
        if algorithm == "hippo":
            result = run_hippo_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "tlbo":
            result = run_tlbo_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "ga":
            result = run_genetic_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "pso":
            result = run_pso_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "sa":
            result = run_sa_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "gwo":
            result = run_gwo_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "hs":
            result = run_hs_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "csa":
            result = run_csa_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "lsa":
            result = run_lsa_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "epo":
            result = run_epo_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "cs":
            result = run_cs_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "abo":
            result = run_abo_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "hem":
            result = run_hem_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "wso":
            result = run_wso_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "de":
            result = run_de_algorithm(pop_size, max_iter, G, T)
        elif algorithm == "abc":
            result = run_abc_algorithm(pop_size, max_iter, G, T)
        else:
            return jsonify({"error": f"Invalid algorithm: {algorithm}"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
