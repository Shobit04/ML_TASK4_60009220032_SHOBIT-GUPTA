
# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier

# # Load the dataset
# data = pd.read_csv('brain_stroke.csv')

# # Preprocessing pipeline
# categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
# preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)])

# # Define the model pipeline
# model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# # Assuming you have already defined 'X' and 'y' using your dataset
# X = data.drop('stroke', axis=1)
# y = data['stroke']

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fitting the model with training data
# model.fit(X_train, y_train)

# # Flask app initialization
# app = Flask(__name__)

# # Route for home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extracting input features from the form
#     gender = request.form['gender']
#     age = float(request.form['age'])
#     hypertension = int(request.form['hypertension'])
#     heart_disease = int(request.form['heart_disease'])
#     ever_married = request.form['ever_married']
#     work_type = request.form['work_type']
#     residence_type = request.form['residence_type']
#     avg_glucose_level = float(request.form['avg_glucose_level'])
#     bmi = float(request.form['bmi'])
#     smoking_status = request.form['smoking_status']
    
#     # Creating input data as a DataFrame
#     input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]], 
#                                columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

#     # Predicting using the model
#     prediction = model.predict(input_data)
    
#     # Formatting the prediction result
#     result = "Brain Stroke" if prediction[0] == 1 else "Healthy"
    
#     return render_template('result.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier

# # Load the dataset
# data = pd.read_csv('brain_stroke.csv')

# # Preprocessing pipeline
# categorical_features = ['gender', 'ever_married', 'work_type', 'smoking_status']
# preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)])

# # Define the model pipeline
# model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# # Assuming you have already defined 'X' and 'y' using your dataset
# X = data.drop('stroke', axis=1)
# y = data['stroke']

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Fitting the model with training data
# model.fit(X_train, y_train)

# # Flask app initialization
# app = Flask(__name__)

# # Route for home page
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extracting input features from the form
#     gender = request.form['gender']
#     age = float(request.form['age'])
#     hypertension = int(request.form['hypertension'])
#     heart_disease = int(request.form['heart_disease'])
#     ever_married = request.form['ever_married']
#     work_type = request.form['work_type']
#     avg_glucose_level = float(request.form['avg_glucose_level'])
#     bmi = float(request.form['bmi'])
#     smoking_status = request.form['smoking_status']
    
#     # Creating input data as a DataFrame
#     input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status]], 
#                                columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

#     # Predicting using the model
#     prediction = model.predict(input_data)
    
#     # Formatting the prediction result
#     result = "Brain Stroke" if prediction[0] == 1 else "Healthy"
    
#     return render_template('result.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('brain_stroke.csv')

# Preprocessing pipeline
categorical_features = ['gender', 'ever_married', 'work_type', 'smoking_status']
preprocessor = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))])

# Assuming you have already defined 'X' and 'y' using your dataset
X = data.drop('stroke', axis=1)
y = data['stroke']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model with training data
model.fit(X_train, y_train)

# Flask app initialization
app = Flask(__name__)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input features from the form
    gender = request.form['gender']
    age = float(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = request.form['ever_married']
    work_type = request.form['work_type']
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = request.form['smoking_status']
    
    # Creating input data as a DataFrame
    input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, ever_married, work_type, avg_glucose_level, bmi, smoking_status]], 
                               columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

    # Predicting using the model
    prediction = model.predict(input_data)
    
    # Formatting the prediction result
    result = "Brain Stroke" if prediction[0] == 0 else "Healthy"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
