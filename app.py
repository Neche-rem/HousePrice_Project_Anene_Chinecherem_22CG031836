from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/house_price_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        try:
            OverallQual = float(request.form['OverallQual'])
            GrLivArea = float(request.form['GrLivArea'])
            TotalBsmtSF = float(request.form['TotalBsmtSF'])
            GarageCars = float(request.form['GarageCars'])
            FullBath = float(request.form['FullBath'])
            YearBuilt = float(request.form['YearBuilt'])

            input_data = np.array([[OverallQual, GrLivArea, TotalBsmtSF,
                                    GarageCars, FullBath, YearBuilt]])

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

        except:
            prediction = "Invalid input, please check your values."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
