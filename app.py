from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib
import traceback
import logging
import numpy as np

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cargar modelos
try:
    encoder_sex = joblib.load('models/encoder_sex.pkl')
    encoder_embarked = joblib.load('models/encoder_embarked.pkl')
    encoder_deck = joblib.load('models/encoder_deck.pkl')
    scaler = joblib.load('models/scaler.pkl')
    pca_model = joblib.load('models/pca_model.pkl')
    knn_model = joblib.load('models/knn_model.pkl')
    
    # Características seleccionadas según el código de Jupyter (#24)
    selected_features = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']
    
    logger.info("Modelos cargados exitosamente")
    
    # Diagnóstico de los encoders
    logger.info(f"Encoder sex feature names: {encoder_sex.get_feature_names_out(['Sex'])}")
    logger.info(f"Encoder embarked categories: {encoder_embarked.categories_}")
    logger.info(f"Encoder deck categories: {encoder_deck.categories_}")
    
except Exception as e:
    logger.error(f"Error cargando modelos: {e}")
    raise e

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraer datos del formulario
        data = {
            'Pclass': int(request.form.get('pclass')),
            'Sex': request.form.get('sex'),
            'Age': float(request.form.get('age')),
            'SibSp': int(request.form.get('sibsp')),
            'Parch': int(request.form.get('parch')),
            'Fare': float(request.form.get('fare')),
            'Embarked': request.form.get('embarked'),
            'Cabin': request.form.get('cabin', 'U')[0].upper() if request.form.get('cabin') else 'U',
            'Name': request.form.get('name', 'Unknown')
        }
        
        logger.info(f"=== DATOS RECIBIDOS ===")
        logger.info(f"Datos: {data}")
        
        # Validaciones básicas
        if data['Age'] < 0 or data['Age'] > 120:
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        
        if data['Fare'] < 0:
            return jsonify({'error': 'Fare must be positive'}), 400
        
        # Crear DataFrame inicial siguiendo el orden del código de Jupyter
        df = pd.DataFrame([{
            'Pclass': data['Pclass'],
            'Sex': data['Sex'],
            'Age': data['Age'],
            'SibSp': data['SibSp'],
            'Parch': data['Parch'],
            'Fare': data['Fare'],
            'Embarked': data['Embarked'],
            'Cabin': data['Cabin']
        }])
        
        logger.info(f"=== DATAFRAME INICIAL ===")
        logger.info(f"Columnas: {df.columns.tolist()}")
        logger.info(f"Valores: {df.iloc[0].to_dict()}")
        
        # Paso 1: Codificar Sex (OneHotEncoder) - Código #6
        encoded_sex = encoder_sex.transform(df[['Sex']])
        encoded_cols = encoder_sex.get_feature_names_out(['Sex'])
        
        # Remover columna Sex y agregar las codificadas
        df = df.drop(columns=['Sex'])
        df[encoded_cols] = encoded_sex
        
        logger.info(f"=== DESPUÉS DE CODIFICAR SEX ===")
        logger.info(f"Columnas: {df.columns.tolist()}")
        logger.info(f"Valores: {df.iloc[0].to_dict()}")
        
        # Paso 2: Codificar Embarked (OrdinalEncoder) - Código #7
        df[['Embarked']] = encoder_embarked.transform(df[['Embarked']])
        
        logger.info(f"=== DESPUÉS DE CODIFICAR EMBARKED ===")
        logger.info(f"Embarked: {df['Embarked'].iloc[0]}")
        
        # Paso 3: Codificar Cabin (OrdinalEncoder) - Código #9
        df[['Cabin']] = encoder_deck.transform(df[['Cabin']])
        
        logger.info(f"=== DESPUÉS DE CODIFICAR CABIN ===")
        logger.info(f"Cabin: {df['Cabin'].iloc[0]}")
        logger.info(f"Todas las columnas: {df.columns.tolist()}")
        logger.info(f"Valores finales: {df.iloc[0].to_dict()}")
        
        # Paso 4: Seleccionar características según código #24
        # Verificar que tenemos todas las características necesarias
        missing_features = [f for f in selected_features if f not in df.columns]
        if missing_features:
            logger.error(f"Faltan características: {missing_features}")
            logger.error(f"Columnas disponibles: {df.columns.tolist()}")
            return jsonify({'error': f'Missing features: {missing_features}'}), 500
        
        # Seleccionar solo las características necesarias
        X = df[selected_features]
        
        logger.info(f"=== CARACTERÍSTICAS SELECCIONADAS ===")
        logger.info(f"Features: {selected_features}")
        logger.info(f"DataFrame X: {X.iloc[0].to_dict()}")
        
        # Paso 5: Aplicar StandardScaler - Código #19
        X_scaled = scaler.transform(X)
        
        logger.info(f"=== DESPUÉS DE SCALER ===")
        logger.info(f"Datos escalados: {X_scaled[0]}")
        
        # Paso 6: Aplicar PCA - Código #27
        X_pca = pca_model.transform(X_scaled)
        
        logger.info(f"=== DESPUÉS DE PCA ===")
        logger.info(f"Datos PCA: {X_pca[0]}")
        logger.info(f"Componentes PCA: {X_pca.shape[1]}")
        
        # Paso 7: Hacer predicción con KNN - Código #28-29
        prediction = knn_model.predict(X_pca)[0]
        probability = knn_model.predict_proba(X_pca)[0]
        
        logger.info(f"=== PREDICCIÓN FINAL ===")
        logger.info(f"Predicción: {prediction}")
        logger.info(f"Probabilidades: {probability}")
        
        return jsonify({
            'survived': bool(prediction),
            'passenger_name': data['Name'],
            'probability': {
                'not_survived': float(probability[0]),
                'survived': float(probability[1])
            },
            'debug_info': {
                'original_features': X.iloc[0].to_dict(),
                'scaled_features': X_scaled[0].tolist(),
                'pca_components': X_pca[0].tolist(),
                'selected_features': selected_features
            }
        })
    
    except Exception as e:
        logger.error(f"Error en /predict: {e}", exc_info=True)
        return jsonify({'error': f'Error en el servidor: {str(e)}'}), 500

@app.route('/test_model', methods=['GET'])
def test_model():
    """Endpoint para probar el modelo con datos conocidos"""
    try:
        # Datos de prueba - caso mujer joven, primera clase
        test_data = pd.DataFrame([{
            'Pclass': 1,
            'Sex': 'female',
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 100.0,
            'Embarked': 'S',
            'Cabin': 'A'
        }])
        
        # Procesar siguiendo los mismos pasos
        # Codificar Sex
        encoded_sex = encoder_sex.transform(test_data[['Sex']])
        encoded_cols = encoder_sex.get_feature_names_out(['Sex'])
        test_data = test_data.drop(columns=['Sex'])
        test_data[encoded_cols] = encoded_sex
        
        # Codificar Embarked
        test_data[['Embarked']] = encoder_embarked.transform(test_data[['Embarked']])
        
        # Codificar Cabin
        test_data[['Cabin']] = encoder_deck.transform(test_data[['Cabin']])
        
        # Seleccionar características
        X = test_data[selected_features]
        
        # Escalar
        X_scaled = scaler.transform(X)
        
        # PCA
        X_pca = pca_model.transform(X_scaled)
        
        # Predicción
        prediction = knn_model.predict(X_pca)[0]
        probability = knn_model.predict_proba(X_pca)[0]
        
        return jsonify({
            'test_case': 'Female, 1st class, age 25',
            'survived': bool(prediction),
            'probability': {
                'not_survived': float(probability[0]),
                'survived': float(probability[1])
            },
            'processed_data': X.iloc[0].to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error en test_model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)