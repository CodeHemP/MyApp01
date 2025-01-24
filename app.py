
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("ann_model.h5")

app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>ANN Prediction</h1>
    <form action="/predict" method="post">
        <label for="x1">Enter x1:</label>
        <input type="number" id="x1" name="x1" required><br><br>
        <label for="x2">Enter x2:</label>
        <input type="number" id="x2" name="x2" required><br><br>
        <button type="submit">Predict</button>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    x1 = float(request.form['x1'])
    x2 = float(request.form['x2'])
    
    # Prepare input for model
    input_data = np.array([[x1, x2]])
    prediction = model.predict(input_data)
    
    # Convert prediction to class (binary classification)
    output = 1 if prediction[0][0] > 0.5 else 0
    return f"<h2>Predicted Output: {output}</h2>"

if __name__ == "__main__":
    app.run()
