import pandas as pd
from sklearn import preprocessing
from joblib import load
from sklearn.preprocessing import StandardScaler

def predict(input: pd.DataFrame) -> pd.DataFrame:
    regr = load('regr.joblib')
    X = input.iloc[:, :].values
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    y_pred = regr.predict(X_scaled)
    return pd.DataFrame({
        'Prediction' : y_pred,
        'HighBP' : input['HighBP'],
        'HighChol' : input['HighChol'],
        'CholCheck' : input['CholCheck'],
        'BMI' : input['BMI'],
        'Smoker' : input['Smoker'],
        'Stroke' : input['Stroke'],
        'HeartDiseaseorAttack' : input['HeartDiseaseorAttack'],
        'PhysActivity' : input['PhysActivity'],
        'Fruits' : input['Fruits'],
        'Veggies' : input['Veggies'],
        'HvyAlcoholConsump' : input['HvyAlcoholConsump'],
        'AnyHealthcare' : input['AnyHealthcare'],
        'NoDocbcCost' : input['NoDocbcCost'],
        'GenHlth' : input['GenHlth'],
        'MentHlth' : input['MentHlth'],
        'PhysHlth' : input['PhysHlth'],
        'DiffWalk' : input['DiffWalk'],
        'Sex' : input['Sex'],
        'Age' : input['Age'],
        'Education' : input['Education'],
        'Income' : input['Income'],
})

def get_output_schema() -> pd.DataFrame:
    return pd.DataFrame({
        'Prediction' : prep_int(),
        'HighBP' : prep_int(),
        'HighChol' : prep_int(),
        'CholCheck' : prep_int(),
        'BMI' : prep_int(),
        'Smoker' : prep_int(),
        'Stroke' : prep_int(),
        'HeartDiseaseorAttack' : prep_int(),
        'PhysActivity' : prep_int(),
        'Fruits' : prep_int(),
        'Veggies' : prep_int(),
        'HvyAlcoholConsump' : prep_int(),
        'AnyHealthcare' : prep_int(),
        'NoDocbcCost' : prep_int(),
        'GenHlth' : prep_int(),
        'MentHlth' : prep_int(),
        'PhysHlth' : prep_int(),
        'DiffWalk' : prep_int(),
        'Sex' : prep_int(),
        'Age' : prep_int(),
        'Education' : prep_int(),
        'Income' : prep_int(),
})