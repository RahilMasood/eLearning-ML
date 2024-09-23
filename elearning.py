import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import rbf_kernel


# Loading the dataset
data = pd.read_csv('Dataset.csv',index_col=0)

# Pattern Identification (Add a column based on total_posts, timeonline, and post types)
def identify_learning_pattern(row):
    if row['helpful_post'] > 10 and row['nice_code_post'] > 5:
        return 'Collaborative'
    elif row['creative_post'] > 5 and row['amazing_post'] > 3:
        return 'Creative'
    elif row['confused_post'] > 5 or row['bad_post'] > 3:
        return 'Confused'
    else:
        return 'Neutral'

print(type(data))

# Applying pattern identification and creating a new column 'learning_pattern'
data['learning_pattern'] = data.apply(identify_learning_pattern, axis=1)
data.replace(',', '.', regex = True, inplace = True)

# Data Preprocessing
X = data.drop(['Approved', 'learning_pattern'], axis=1)  # Features (excluding 'Approved' and 'learning_pattern')
y = data['Approved']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Checking for missing values
data.isnull().sum()

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Checking for any remaining non-numeric columns and removing them
data = data.select_dtypes(include=[np.number])

# Scaling the data
scaler = StandardScaler()
# Creating a DataFrame
data_scaled = scaler.fit_transform(data.drop(['Approved'], axis=1))

# Quadratic Support Vector Machine (SVM)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=7)

# Training the Quadratic SVM model (Polynomial kernel with degree=2)
svm_model = SVC(kernel='poly', degree=2, C=1.0)  # Quadratic SVM
svm_model.fit(X_train, y_train)

# Making predictions and evaluateing the model
y_pred = svm_model.predict(X_test)

# Evaluating the model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
