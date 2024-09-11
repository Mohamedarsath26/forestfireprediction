#10/9/24

from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
application = Flask(__name__)
app = application

# Load pre-trained model and scaler
ridge_model = pickle.load(open("Linear_regression/End to end ml project/model.pkl", 'rb'))
scaler = pickle.load(open("Linear_regression/End to end ml project/scaler.pkl", 'rb'))

# Home route to render the HTML form
@app.route("/")
def index():
    return render_template('home.html')

# Prediction route
@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Extract form data from POST request
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale the input data
            new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Predict using the model
            result = ridge_model.predict(new_data)
            print(result)
            # Render the template and show the result
            return render_template('home.html', results=result[0])

        except Exception as e:
            # Handle any error, log it, and display error message
            print(f"Error during prediction: {e}")
            return render_template('home.html', results="Error in prediction. Please check the input data.")
    else:
        return render_template('home.html')

# Start the app
if __name__ == "__main__":
    app.run(host='0.0.0.0')
