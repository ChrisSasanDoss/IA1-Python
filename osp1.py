import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time

df = pd.read_csv('/content/DSAI-LVA-DATASET for Quiz.csv')

for i, rows in df.iterrows():
    if rows['Pass'] == 'Yes' and rows['PreviousTestScore'] >= 75:
        df.loc[i, 'Result'] = 'HighPass'
    elif rows['Pass'] == 'Yes' and rows['PreviousTestScore'] < 75:
        df.loc[i, 'Result'] = 'LowPass'
    elif rows['Pass'] == 'No':
        df.loc[i, 'Result'] = 'Fail'

df = df.drop('Pass', axis=1)

parent_edu = ['Masters', 'Bachelor''s', 'College', 'High School', 'Not Educated']
df['Parent_Education'] = np.random.choice(parent_edu, size=len(df['StudyTime']))

df = df.drop('ParentEducation', axis=1)

df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.7 * len(df_shuffled))
train_set = df_shuffled.iloc[:train_size]
test_set = df_shuffled.iloc[train_size:]

test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')

lbl = LabelEncoder()
train_df['Parent_Education'] = lbl.fit_transform(train_df['Parent_Education'])
test_df['Parent_Education'] = lbl.transform(test_df['Parent_Education'])
train_df['Result'] = lbl.fit_transform(train_df['Result'])
test_df['Result'] = lbl.transform(test_df['Result'])

X_train = train_df.drop('Result', axis=1)
X_test = test_df.drop('Result', axis=1)
y_train = train_df['Result']
y_test = test_df['Result']

model_name = [
    ('Decision Tree Classifier', DecisionTreeClassifier()),
    ('K Nearest Neighbors', KNeighborsClassifier(n_neighbors=2)),
    ('SVM', SVC()),
    ('XGB Classifier', XGBClassifier(learning_rate=0.01, gamma=3))
]

results = {}

for name, model in model_name:
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    time_taken = round((time.time() - start_time), 2)
    accuracy = accuracy_score(y_pred, y_test)
    results[name] = accuracy
    print(f'{name} \nAccuracy: {accuracy * 100:.2f}% \nTime Taken: {time_taken} sec\n')

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'bar'}]])

x_values = list(results.keys())
y_values = list(results.values())

fig.add_trace(go.Bar(x=x_values, y=y_values, marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)',width=1.5))), row=1, col=1)

fig.update_layout(title='Model Accuracy Comparison', xaxis_title='Model', yaxis_title='Accuracy', hovermode='x')

fig.show()
