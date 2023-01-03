import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import SGDClassifier
from pprint import pprint

clf = SGDClassifier(loss='hinge', learning_rate='constant', eta0=0.01, random_state=42)
data = pd.read_csv('dataset.csv', skipinitialspace = True)
disease_file = pd.read_csv('Disease_Description.csv', skipinitialspace = True)
symptom_file = pd.read_csv('Symptom-severity.csv', skipinitialspace = True)
df = pd.DataFrame(data)
df.drop_duplicates(inplace = True)
df.fillna('None', inplace = True)

disease_unique = disease_file['Disease']
symptoms_unique = symptom_file['Symptom']
symptoms_unique = symptoms_unique.tolist()

symptom_list = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8',
                'Symptom_9', 'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
                'Symptom_16', 'Symptom_17']

X = df[symptom_list]
y = df['Disease']

encoder = OneHotEncoder(handle_unknown = 'ignore')
encoder.fit(X)
X_encoded = encoder.transform(X)
X = X_encoded
# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a KNN model using the training set
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Use the trained KNN model to make a prediction on the test set
predictions = knn.predict(X_test)

# Evaluate the model's performance on the test set
print("Symptoms List: ")
pprint(symptoms_unique)
# Use the trained KNN model to make a diagnosis for a new patient with the following symptoms
symptoms = ['None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None', 'None',
            'None', 'None', 'None', 'None']
i = 0
n = 0
while i < 17:
    symptom = input('Enter one symptom from the "Symptoms List" above (if there is no more then enter "Done":) ')
    if symptom.lower() == "done":
        break
    elif symptom.lower() in symptoms_unique:
        symptoms[n] = symptom
        n += 1

symptoms_encoded = encoder.transform([symptoms])
prediction = knn.predict(symptoms_encoded)
print('Diagnosis:', prediction)
symptoms_listed = []
for n in symptoms:
    if n != 'None':
        symptoms_listed.append(n)
print(f'The diagnosis was made based on these symptoms: {symptoms_listed}')

accuracy = accuracy_score(y_test, predictions)
print('Accuracy:', accuracy)