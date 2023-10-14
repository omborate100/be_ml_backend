from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app,origins='http://localhost:3000')
## load pickle model
model = pickle.load(open("model.pkl","rb"))
model1=pickle.load(open('model1.pkl','rb'))

@app.route('/')
def index():
    # return render_template('index.html')
    return "Sucessfully deployed!!! "

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    no_city = data.get('city_expansion')
    market_size = data.get('market_size')
    sales_prev_years =data.get('sale_prev_year')
    sales_prev_month = data.get('sale_prev_month')
    profit = data.get('profit')
    ebidta = data.get('ebidta')
    gross_margin = data.get('gross_margin')
    amount_for_equity = data.get('amount_for_equity')
    equity_ask = data.get('equity_ask')
    valuation = data.get('valuation')
    features = [ no_city, market_size, sales_prev_years, sales_prev_month, profit,ebidta, gross_margin, amount_for_equity,equity_ask, valuation]
    
    
    float_features = [float(x) for x in features]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    val = prediction[0]    
    
    # float_feature = [float(x) for x in features1]
    # features = [np.array(features)]
    # prediction = model.predict(features)
    # print(prediction)
    # val = prediction[0]
    print(val)
    # result = "Startup will become successful in coming years" if prediction[0] == 1 else "Startup will not become successful in coming years"
    if prediction[0] == 1: 
        result = "Startup will become successful";
    else:
        result = "Startup will not become successful"    

    # print(result)
    return result
    if val == True:
        return render_template('index.html' , value = "Startup will become successful in coming years")
    else:
        return render_template('index.html' , value = "Startup will not become successful in coming years")
        
@app.route('/probability', methods = ['POST'])
def prob():
    data = request.get_json()
    no_city = data.get('city_expansion')
    market_size = data.get('market_size')
    sales_prev_years =data.get('sale_prev_year')
    sales_prev_month = data.get('sale_prev_month')
    profit = data.get('profit')
    ebidta = data.get('ebidta')
    gross_margin = data.get('gross_margin')
    amount_for_equity = data.get('amount_for_equity')
    equity_ask = data.get('equity_ask')
    valuation = data.get('valuation')
    features = [ no_city, market_size, sales_prev_years, sales_prev_month, profit,ebidta, gross_margin, amount_for_equity,equity_ask, valuation]
    
    float_features = [float(x) for x in features]
    final=[np.array(float_features)]
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    print(type(output))
    output = float(output)
    print(type(output))
    ans=output*100
    type(jsonify(ans))
    return jsonify(ans)

if __name__ == '__main__':
    app.run(debug=True)
    
    
