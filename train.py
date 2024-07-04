import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open("data.pickle", 'rb'))


data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])


train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42, stratify=labels)

model = RandomForestClassifier()
model.fit(train_data, train_labels)


y_pred = model.predict(test_data)

print("Accuracy: ", round(accuracy_score(test_labels, y_pred), 4))

f = open("model.pickle", 'wb')
pickle.dump({'model': model}, f)
f.close()