
# ## Load the Model

import pickle
import os

print('Loading the model up....')
model_file = f'{os.getcwd()}/model_C={1.0}.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


from flask import Flask
from flask import request
from flask import jsonify

app = Flask('churn')


@app.route('/predict', methods =['POST'])


def predict():
    customer = request.get_json()
    print(f'Customer: {customer}')
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return result

if __name__ == '__main__':
    app.run(debug=True, )