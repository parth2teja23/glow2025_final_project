import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (mean_squared_error, accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv('winequality-red.csv')

# 2. EDA
print(df.info(), df.describe())
sns.histplot(data=df, x='quality', bins=range(3,10))
plt.show()

# 3. Correlation
corr = df.corr()
sns.heatmap(corr, annot=True)
plt.show()

# 4. Preprocess
X = df.drop('quality', axis=1)
y = df['quality']
y_bin = (y >= 7).astype(int)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_bin, test_size=0.2, random_state=42)
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train, y_train)

# 5. Train RF classifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_bal, y_train_bal)

# 6. Evaluate
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print('Confusion matrix:', confusion_matrix(y_test, y_pred))
