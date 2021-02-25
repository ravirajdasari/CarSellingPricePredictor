import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    print(features)
    int_features = [float(x) for x in features[:6]]
    onehotvalues = []
    cat_features = [str(x) for x in features[6:]]

    if cat_features[0] == "cng":
        onehotvalues.extend([1,0,0,0])
    elif cat_features[0] == "diesel":
        onehotvalues.extend([0,1,0,0])
    elif cat_features[0] == "lpg":
        onehotvalues.extend([0,0,1,0])
    else :
        onehotvalues.extend([0,0,0,1])

    if cat_features[1] == "dealer":
        onehotvalues.extend([1,0,0])
    elif cat_features[1] == "individual":
        onehotvalues.extend([0,1,0])
    else :
        onehotvalues.extend([0,0,1])

    if cat_features[2] == "automatic":
        onehotvalues.extend([1,0])
    else:
        onehotvalues.extend([0,1])

    #1,4,2,test,3
    if cat_features[3] == "first":
        onehotvalues.extend([1,0,0,0,0])
    elif cat_features[3] == "fourth and above":
        onehotvalues.extend([0,1,0,0,0])
    elif cat_features[3] == "second":
        onehotvalues.extend([0,0,1,0,0])
    elif cat_features[3] == "test drive car":
        onehotvalues.extend([0,0,0,1,0])
    else :
        onehotvalues.extend([0,0,0,0,1])
    int_features.extend(onehotvalues)
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Predicted Selling Price  is Rs. {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
