from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    
    # Convert to array
    final_features = np.array([features])
    
    # Predict (THIS WAS MISSING ❗)
    prediction = model.predict(final_features)

    # Convert USD to INR
    price_inr = prediction[0] * 83

    return render_template('index.html',
                           prediction_text=f'Predicted Price: ₹ {price_inr:,.2f}')
# Run app
if __name__ == "__main__":
    app.run(debug=True)