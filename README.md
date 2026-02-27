# Dogs vs Cats Classifier (62% Accuracy)

**Production ready cat/dog classification on real images using scikit-learn RandomForestClassifier!**

[![Accuracy](https://img.shields.io/badge/Accuracy-62%25-yellow)](https://www.kaggle.com/datasets/marquis03/cats-and-dogs)
[![Dataset](https://img.shields.io/badge/Dataset-Real%20images-blue)](https://www.kaggle.com/datasets/marquis03/cats-and-dogs)
[![Python](https://img.shields.io/badge/Python-3.13-orange)](https://www.python.org/)

## 📌 Results
| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **62.0%** |
| **Image Size** | **32x32 pixels** |
| **Features** | **3072 flattened pixels** |
| **Model** | **RandomForestClassifier** |
| **Test Images** | **70 validation** |

## 📁 Dataset
[Kaggle Cats and Dogs Dataset](https://www.kaggle.com/datasets/marquis03/cats-and-dogs)  
**train/cat/** & **train/dog/** (training)  
**val/cat/** & **val/dog/** (validation)  

**Note**: 62% baseline beats random guessing (50%) on real images using pixel features + RandomForest. CNN would hit 90%+.
