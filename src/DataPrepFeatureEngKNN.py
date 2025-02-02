from keras.datasets import mnist
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd 
from skimage.feature import hog 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt

mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalizing the input data 

x_train_norm = x_train/255.0
x_test_norm = x_test/255.0

# extract the features using HOG 

def feature_extractor(dataset):
    hog_features = []
    for data in dataset:
        fd = hog(data, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                 block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True)
        hog_features.append(fd)
    return np.array(hog_features)

x_train_features = feature_extractor(x_train_norm)
x_test_features = feature_extractor(x_test_norm)
    

x_train_features = pd.DataFrame(x_train_features)
x_test_features = pd.DataFrame(x_test_features)

# print("HOG features for first sample:", x_train_features[0])  
# print("Shape of HOG features (x_train):", x_train_features.shape)
# print("Shape of HOG features (x_test):", x_test_features.shape)


scaler = StandardScaler()
x_train_stand = scaler.fit_transform(x_train_features)
x_test_stand = scaler.transform(x_test_features)


# baseline multi class logreg just to check the hog features

# logreg = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', max_iter = 1000).fit(x_train_stand, y_train)
# print(logreg.coef_) # coef of the weights for the features
# score = logreg.score(x_test_stand, y_test)
# print("mean accuracy:", score)

# knn


# for cv =5 , best neighbor is 6, for cv = 10, best neighbor is still 6 
# for cv = 3, best neighbor is 4 


# for cv = 3, best neighbor = 4, best_score = 0.9432166666666667

# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors': np.arange(1, 20)}
# knn_gscv = GridSearchCV(knn, param_grid, cv=3)
# knn_gscv.fit(x_train_stand, y_train)
# print(knn_gscv.best_score_)

# for cv = 5, best neighbor = 6, best_score = 0.9459

# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors': np.arange(1, 20)}
# knn_gscv = GridSearchCV(knn, param_grid, cv=5)
# knn_gscv.fit(x_train_stand, y_train)
# print(knn_gscv.best_score_)

# for cv = 10, best neighbor = 6, best_score = 0.9470333333333333

# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors': np.arange(1, 20)}
# knn_gscv = GridSearchCV(knn, param_grid, cv=10)
# knn_gscv.fit(x_train_stand, y_train)
# print(knn_gscv.best_score_)

knn = KNeighborsClassifier(n_neighbors=6)
skf = StratifiedKFold(n_splits=10)
skf_scores=[]
for train_index, val_index in skf.split(x_train_stand, y_train):
    x_train_fold, x_val_fold = x_train_stand[train_index], x_train_stand[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    knn.fit(x_train_stand, y_train)
    skf_scores.append(knn.score(x_val_fold, y_val_fold))
print(f"Cross-Validation mean accuracy (k=10): {np.mean(skf_scores)}")

# evaluating on the test set 

y_pred = knn.predict(x_test_stand)
test_accuracy = knn.score(x_test_stand, y_test)
print(f"Test accuracy: {test_accuracy}")
print(classification_report(y_test, y_pred))
