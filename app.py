from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import logging

app = Flask(__name__)

# Cargar modelos
encoder_sex = joblib.load('models/encoder_sex.pkl')
encoder_deck = joblib.load('models/encoder_deck.pkl')
scaler = joblib.load('models/scaler.pkl')
pca_model = joblib.load('models/pca_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')

# Solo estas características fueron usadas
selected_features = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        data = {
            'Pclass': int(request.form['pclass']),
            'Sex': request.form['sex'],
            'Age': float(request.form['age']),
            'Fare': float(request.form['fare']),
            'Cabin': request.form['cabin'][0].upper()
        }

        df = pd.DataFrame([data])

        # Codificar 'Sex'
        encoded_sex = encoder_sex.transform(df[['Sex']])
        encoded_cols = encoder_sex.get_feature_names_out(['Sex'])
        df.drop(columns=['Sex'], inplace=True)
        df[encoded_cols] = encoded_sex

        # Codificar 'Cabin'
        df[['Cabin']] = encoder_deck.transform(df[['Cabin']])

        # Seleccionar las características exactas
        X = df[selected_features]

        # Escalar
        X_scaled = scaler.transform(X)

        # Reducir con PCA
        X_pca = pca_model.transform(X_scaled)

        # Predecir
        prediction = knn_model.predict(X_pca)[0]
        probabilities = knn_model.predict_proba(X_pca)[0]

        return jsonify({
            'survived': bool(prediction),
            'probability': {
                'not_survived': float(probabilities[0]),
                'survived': float(probabilities[1])
            }
        })

    except Exception as e:
        logging.exception("Error en /predict")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
