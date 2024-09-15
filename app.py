from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('decision_tree_model.pkl')

features = ['Age', 'BusinessTravel', 'EnvironmentSatisfaction', 'JobRole', 
            'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked', 
            'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = {feature: request.form[feature] for feature in features}
    
    user_input['Age'] = int(user_input['Age'])
    user_input['EnvironmentSatisfaction'] = int(user_input['EnvironmentSatisfaction'])
    user_input['JobSatisfaction'] = int(user_input['JobSatisfaction'])
    user_input['MonthlyIncome'] = float(user_input['MonthlyIncome'])
    user_input['NumCompaniesWorked'] = int(user_input['NumCompaniesWorked'])
    user_input['TotalWorkingYears'] = int(user_input['TotalWorkingYears'])
    user_input['WorkLifeBalance'] = int(user_input['WorkLifeBalance'])
    user_input['YearsAtCompany'] = int(user_input['YearsAtCompany'])
    
    input_df = pd.DataFrame([user_input])
    input_df = pd.get_dummies(input_df)
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    
    prediction = model.predict(input_df)
    
    
    result = "The employee is likely to stay." if prediction[0] == 0 else "The employee is likely to leave."
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
