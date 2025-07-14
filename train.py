#1
import pandas as pd
df: pd.DataFrame = pd.read_csv("titanic_train.csv")
df.head()

#2
df.info()

#3
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
df.head()

#4
no_nuls = df.isnull().sum()
no_nuls[no_nuls > 0]

#5
from sklearn.impute import SimpleImputer

for_mode = ['Embarked']
for_new_class = ['Cabin']

imputer_mode = SimpleImputer(strategy='most_frequent')
df[for_mode] = imputer_mode.fit_transform(df[for_mode])


imputer_new_class = SimpleImputer(strategy='constant', fill_value='Unknown')
df[for_new_class] = imputer_new_class.fit_transform(df[for_new_class])

no_nuls = df.isnull().sum()
no_nuls[no_nuls > 0]

#6
from sklearn.preprocessing import OneHotEncoder
import numpy as np

for_sex = ['Sex']

encoder_sex = OneHotEncoder(drop=None, sparse_output=False)

encoded_sex = encoder_sex.fit_transform(df[for_sex])
encoded_cols = encoder_sex.get_feature_names_out(for_sex)

df.drop(columns=for_sex, inplace=True)
df[encoded_cols] = encoded_sex

df.head()

#7
from sklearn.preprocessing import OrdinalEncoder

embarked_order = [['S', 'C', 'Q']]
for_embarked = ['Embarked']
encoder_embarked = OrdinalEncoder(categories=embarked_order)
df[for_embarked] = encoder_embarked.fit_transform(df[for_embarked])

df.head()

#8
df['Cabin'] = df['Cabin'].str[0]
df.head()

#9
from sklearn.preprocessing import OrdinalEncoder

deck_order = [['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'U']]
for_deck = ['Cabin']
encoder_deck = OrdinalEncoder(categories=deck_order)
df[for_deck] = encoder_deck.fit_transform(df[for_deck])

df.head()


#10
df_know_age = df[df['Age'].notnull()]
df_know_age.head()

#11
df_unknown_age = df[df['Age'].isnull()]
df_unknown_age.head()

#12
X = df_know_age.drop(columns=['Age'])
y= df_know_age['Age']

#13
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X, y)

#14
features_importance: pd.DataFrame = pd.DataFrame({
    'feature': X.columns,
    'importance': dtr.feature_importances_ * 100
}).sort_values(by='importance', ascending=False)
features_importance

#15
features_for_age = ['Fare', 'Parch', 'Pclass', 'SibSp', 'Cabin']
X_train: pd.DataFrame = df_know_age[features_for_age]
y_train: pd.Series = df_know_age['Age']
X_test: pd.DataFrame = df_unknown_age[features_for_age]

#16
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

#17
y_predict = model.predict(X_test) 

#18
df.loc[df['Age'].isnull(), 'Age'] = y_predict
df.head()

#19
from sklearn.preprocessing import StandardScaler
# Usar solo las columnas finales seleccionadas
cols_selected = ['Sex_female', 'Age', 'Fare', 'Pclass', 'Cabin']
scaler = StandardScaler()
df[cols_selected] = scaler.fit_transform(df[cols_selected])
df.head()

#20
X = df.drop(columns=['Survived'])
X.head()

#21
y = df['Survived']
y

#22
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X, y)

#23
features_importance: pd.DataFrame = pd.DataFrame({
    'feature': X.columns,
    'importance': dtc.feature_importances_ * 100
}).sort_values(by='importance', ascending=False)

features_importance

#24
features = ['Sex_female', 'Age', 'Fare', 'Pclass',"Cabin"]
X = X[features]
X.head()

#25
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(
  X, 
  y, 
  train_size=0.8, 
  random_state=42,
  shuffle=True,
)

#26
# En una nueva celda después del train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Ajustar PCA con todos los componentes para ver la varianza
pca_analysis = PCA().fit(X_train)

# Graficar la varianza explicada acumulada
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca_analysis.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada por Componentes Principales')
plt.grid(True)
# Línea para el 95% de varianza
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Varianza')
plt.legend(loc='best')
plt.show()

#27
from sklearn.decomposition import PCA

pca = PCA(n_components=0.90, random_state=42)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


#28
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train_pca, y_train)

#29
y_pred = model.predict(X_test_pca)

#30
from sklearn.metrics import accuracy_score
precision = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {precision}%")



# Save models
import joblib
import os
os.makedirs('models', exist_ok=True)
joblib.dump(encoder_sex, 'models/encoder_sex.pkl')
joblib.dump(encoder_embarked, 'models/encoder_embarked.pkl')
joblib.dump(encoder_deck, 'models/encoder_deck.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca_model.pkl')
joblib.dump(model, 'models/knn_model.pkl')

print("Models trained and saved successfully.")
