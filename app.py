import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load your model and columns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model/model.pkl"), "rb"))
model_columns = pickle.load(open(os.path.join(BASE_DIR, "model/columns.pkl"), "rb"))

# Professional Bootstrap Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agri-Tech | Crop Prediction & Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar { background-color: #2c3e50; }
        .card { border: none; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .btn-primary { background-color: #3498db; border: none; }
        .math-box { background-color: #ffffff; border-left: 5px solid #2ecc71; }
        .result-header { color: #2c3e50; font-weight: 700; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark mb-4">
        <div class="container">
            <span class="navbar-brand mb-0 h1">Vector Calculus Project: Crop Optimization</span>
        </div>
    </nav>

    <div class="container">
        <div class="row">
            <div class="col-md-5">
                <div class="card p-4">
                    <h4 class="mb-4 result-header">System Inputs</h4>
                    <form action="/predict" method="post">
                        <div class="mb-3">
                            <label class="form-label">Total Area (Hectares)</label>
                            <input type="number" step="any" class="form-control" name="Area" value="{{ inputs.Area or 1000 }}" required>
                        </div>
                        <div class="mb-3">
                            <label class="form-label">Agricultural Year</label>
                            <input type="number" class="form-control" name="Crop_Year" value="{{ inputs.Crop_Year or 2024 }}" required>
                        </div>
                        <button type="submit" class="btn btn-primary w-100">Calculate Model Output</button>
                    </form>
                </div>
            </div>

            <div class="col-md-7">
                {% if prediction %}
                <div class="card p-4 mb-4">
                    <h4 class="result-header">Primary Prediction</h4>
                    <h2 class="text-success">{{ prediction }} <small class="text-muted">Units</small></h2>
                </div>

                <div class="card p-4 math-box">
                    <h5 class="text-primary">Vector Calculus Analysis</h5>
                    <p class="text-muted small">Requirement: Gradient-based learning and system modeling [cite: 4, 7]</p>
                    <hr>
                    <div class="row">
                        <div class="col-sm-6">
                            <h6>Mathematical Operation</h6>
                            <code class="d-block p-2 bg-light rounded">∇f ≈ [f(x + Δh) - f(x)] / Δh</code>
                        </div>
                        <div class="col-sm-6">
                            <h6>Calculated Gradient (Area)</h6>
                            <h4 class="text-dark">{{ gradient }}</h4>
                        </div>
                    </div>
                    <div class="mt-3 p-3 bg-light rounded">
                        <strong>Insight:</strong> This partial derivative represents the sensitivity of the model to the "Area" vector component. For every unit change in land area, production is projected to change by <strong>{{ gradient }}</strong> units.
                    </div>
                </div>
                {% else %}
                <div class="text-center mt-5 text-muted">
                    <h5>Enter parameters and run calculation to see Vector Calculus analysis.</h5>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, inputs={})

@app.route('/predict', methods=['POST'])
def predict():
    raw_data = request.form.to_dict()
    inputs = {k: float(v) for k, v in raw_data.items() if v}
    
    df_input = pd.DataFrame([inputs])
    df_final = pd.get_dummies(df_input).reindex(columns=model_columns, fill_value=0)

    # Base Prediction f(x)
    prediction = model.predict(df_final)[0]

    # Numerical Gradient calculation (Partial Derivative)
    h = 1.0 
    df_h = df_final.copy()
    # Numerical Gradient calculation (Partial Derivative)
    # INCREASE h to 500 or 1000 to bridge the "steps" of the Random Forest
    h = 500.0 
    df_h = df_final.copy()
    if 'Area' in df_h.columns:
        df_h['Area'] += h
        prediction_h = model.predict(df_h)[0]
        gradient_area = (prediction_h - prediction) / h
    else:
        gradient_area = 0.0

    return render_template_string(HTML_TEMPLATE, 
                                 prediction=f"{prediction:,.2f}", 
                                 gradient=f"{gradient_area:.4f}",
                                 inputs=inputs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
