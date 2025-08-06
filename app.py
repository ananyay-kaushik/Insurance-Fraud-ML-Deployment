# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'finalized_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model not available. Try again later.")
    
    try:
        # Extract form data
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Fraud' if prediction[0] == 1 else 'Not Fraud'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True)

