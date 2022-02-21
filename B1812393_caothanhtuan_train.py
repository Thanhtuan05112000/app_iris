import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv('./iris_dataset.csv')
df = data.iloc[:, 0:4]
y = data.variety

# huan luyen mo hinh
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=100)
# xay dung mo hinh voi k = 5
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Tính độ chính xác
print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
      model.score(X_test, y_test))
# Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: 0.978

# Save model
file = open('./model_knn', 'wb')
pickle.dump(model, file)