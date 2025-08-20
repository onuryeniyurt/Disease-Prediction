from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)
df_precaution=pd.read_csv('data/symptom_precaution.csv')
df_description=pd.read_csv('data/symptom_Description.csv')
df = pd.read_csv('data/df_weighted.csv')
original_symptoms = [col for col in df.columns if col != 'Disease']

display_symptoms = [symptom.replace('_', ' ').strip().capitalize()
                   for symptom in original_symptoms]


weights_df = pd.read_csv('data/weights_df.csv')
weights_df['Symptom'] = weights_df['Symptom'].str.replace('_', ' ').str.strip().str.capitalize()
weights_df["weight"] = weights_df["weight"] / weights_df["weight"].max()


model = tf.keras.models.load_model("my_model.keras")

@app.route("/")
def home():
    return render_template("index.html", symptoms=display_symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    total_features = len(display_symptoms)
    x_input = np.zeros((1, total_features))

    for s in data['symptoms']:
        try:
            idx = display_symptoms.index(s)
            weight = 1.0
            if s in weights_df['Symptom'].values:
                weight = weights_df.loc[weights_df['Symptom'] == s, 'weight'].values[0]
            x_input[0, idx] = weight
        except ValueError:
            continue

    prediction = model.predict(x_input)
    predicted_class = int(np.argmax(prediction, axis=1)[0])


    encoder = LabelEncoder()
    df['Disease_Encoded'] = encoder.fit_transform(df['Disease'])

    predicted_disease = encoder.inverse_transform([predicted_class])[0]
    df_precaution['Disease']=df_precaution['Disease'].str.lower()
    row = df_precaution[df_precaution["Disease"] == predicted_disease.lower()]
    precautions = []
    precautions = row.iloc[0, 1:].dropna().tolist()

    df_description['Disease']=df_description['Disease'].str.lower()
    description=df_description[df_description['Disease']==predicted_disease.lower()]
    description=description.iloc[0,1]
    return jsonify({"prediction": predicted_disease,
                    "precautions": precautions,
                    "description": description})

if __name__ == "__main__":
    app.run(debug=True)