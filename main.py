from flask import Flask, render_template, request
import dill as pickle

import pandas as pd
import numpy as np

app = Flask(__name__)

model = pickle.load(open('LinearRegressionModel4.pkl', 'rb'))

car = pd.read_csv('Cleaned car.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('name')
    year = request.form.get('year')

    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kms_driven')

    # Prepare input for prediction
    prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                             data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))


    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)
