import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
data = {
'Temperature': [30, 25, 27, 20, 21, 23, 26, 30, 35, 28],
'Humidity': [70, 60, 65, 55, 52, 58, 67, 75, 80, 68],
'Windy': [False, True, False, True, False, True, False, True, False,
True],
'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes']
}
df = pd.DataFrame(data)
df['Windy'] = df['Windy'].astype(int)
df['PlayTennis'] = df['PlayTennis'].apply(lambda x: 1 if x == 'Yes' else
0)
X=df[['Temperature', 'Humidity', 'Windy']]
y =df['PlayTennis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
new_data = pd.DataFrame({'Temperature': [22, 32], 'Humidity': [60,
75], 'Windy': [1, 0]})
predictions = clf.predict(new_data)
print("Predictions:", predictions)
