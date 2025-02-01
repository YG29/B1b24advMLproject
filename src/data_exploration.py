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
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

mnist.load_data(path="mnist.npz")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#visualising the data 
digits = np.arange(10)
images = []

for digit in digits:
    indices = np.where(y_train==digit)[0][:2]
    images.extend(x_train[indices])

fig, axes = plt.subplots(4,5, figsize=(5,4))
index = 0
for row in range(4):
    for column in range(5):
        ax = axes[row, column]
        ax.imshow(images[index], cmap = 'gray')
        ax.axis('off')
        index +=1

plt.tight_layout()
plt.show()

#visualising the hog 
unique_digits = np.arange(10)  # Digits 0 to 9
selected_images = []

for digit in unique_digits:
    index = np.where(y_train == digit)[0][0]  
    selected_images.append(x_train[index])  


fig, axes = plt.subplots(5, 4, figsize=(8, 6))

for i, digit in enumerate(unique_digits):
    hog_features, hog_image = hog(selected_images[i], orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys',
                                  visualize=True)


    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))


    row, col = divmod(i, 2)  
    axes[row, col * 2].imshow(selected_images[i], cmap='gray')
    axes[row, col * 2].set_title(f"Digit {digit}")
    axes[row, col * 2].axis("off")


    axes[row, col * 2 + 1].imshow(hog_image_rescaled, cmap='gray')
    axes[row, col * 2 + 1].set_title(f"HOG of {digit}")
    axes[row, col * 2 + 1].axis("off")

plt.tight_layout()
plt.show()