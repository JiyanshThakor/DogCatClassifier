from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import joblib

x_train = []
y_train = []

for folder in ['train/cat','train/dog']:
    label = 0 if os.path.basename(folder) == 'cat' else 1
    for imgfile in os.listdir(folder):
        if imgfile.lower().endswith(('.jpg', '.jpeg')):
            imgpath = os.path.join(folder, imgfile)
            img = Image.open(imgpath)
            imgarray = np.array(img.resize((32, 32))).flatten()
            x_train.append(imgarray)
            y_train.append(label)

x_test = []
y_test = []

for folder in ['val/cat', 'val/dog']:
    label = 0 if os.path.basename(folder) == 'cat' else 1
    for imgfile in os.listdir(folder):
        if imgfile.lower().endswith(('.jpg', '.jpeg')):
            imgpath = os.path.join(folder, imgfile)
            img = Image.open(imgpath)
            imgarray = np.array(img.resize((32, 32))).flatten()
            x_test.append(imgarray)
            y_test.append(label)

model = RandomForestClassifier(n_estimators=60)
x_train_flat = np.array(x_train)
x_val_flat = np.array(x_test)

model.fit(x_train_flat, y_train)
print(accuracy_score(y_test, model.predict(x_val_flat)))  

joblib.dump(model, "DogCatClassifier.pkl")