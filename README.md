# AI_MAGEDDON
THIS IS A HACKATHON PROJECT BY TEAM DR


INTRODUCTION:

INTRUSION DETECTION:

Intrusion detection is the process of identifying malicious activity or unauthorized access in a computer network.

Intrusions can include attacks such as DoS (Denial-of-Service), U2R (User to Root), R2L (Remote to user), PROBE.

There are two main types of intrusion detection: signature-based and anomaly-based.

Intrusion detection is a critical component of network security because it can help identify and respond to security incidents in real-time.

DoS (Denial-of-Service) attacks aim to disrupt the availability of network resources by flooding the network with traffic.

Probe attacks are used to gather information about a network's vulnerabilities and configuration.




Dataset Description:

DATASET: NSL-KDD

The dataset used for intrusion detection research is an improved version of the KDDCup 99 data set, which was widely used as one of the few publicly available data sets for evaluating intrusion detection systems (IDSs) until the release of the NSL-KDD data set. The NSL-KDD data set includes network traffic data from a local area network (LAN) simulation environment that is designed to resemble a typical US Air Force LAN.

The NSL-KDD data set has been preprocessed to eliminate redundancy and inconsistency in the original KDDCup 99 data set.


Redefined Problem Statement:

• Given a network traffic dataset, develop a machine learning model that can accurately classify each network connection as either normal or malicious. The model should be able to identify different types of attacks, such as Denial of Service (DoS), Probe, User to Root (U2R), and Remote to Local (R2L) attacks.



INTRODUCTION Contd...

R2L (Remote-to-Local) attacks target vulnerabilities in the network's authentication and access controls to gain unauthorized access.

U2R (User-to-Root) attacks exploit vulnerabilities in the system to gain root-level access.

We used NSL-KDD dataset, which is a modified version of the KDD Cup 1999 dataset, which is a widely used benchmark dataset for intrusion detection research.

The NSL-KDD dataset includes a preprocessed version of the KDD Cup 1999 dataset that has been normalized and de-duplicated to reduce noise and improve accuracy.

The dataset is widely used in intrusion detection research as a benchmark for evaluating the performance of different machine learning algorithms such as Random Forest, Naive Bayes', KNN(K-Nearest Neighbour), SVM (Support Vector Machines), Decision Trees, Logistic Regression, ANN (Artificial Neural Networks).



DATA DESCRIPTION:

Used two different datasets, one is train dataset and the other is test dataset.

Variables in train dataset are:

"duration": the length of time in seconds that the connection lasted.

"protocol_type": the protocol used for the connection (tcp, udp, etc.).

"service": the type of service being used (ftp_data, http, etc.).

"flag": a flag indicating the status of the connection (SF, SO, REJ, etc.).

"sre bytes": the number of data bytes from the source to the destination in the connection.

"dst_bytes": the number of data bytes from the destination to the source in the connection.

"land": a flag indicating if the connection is from/to the same host/port.

"wrong_fragment": the number of "wrong" fragments in the connection.

"urgent": the number of urgent packets in the connection


• Variables in test dataset are:

Protocol type (e.g., tep, icmp)

Service (e.g., ftp_data, http, eco_i)

Flag (e.g., SF, REJ, RSTO)

Various numerical features (e.g., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)

Attack type (e.g., neptune, normal, saint, mscan)
Shape of Training Dataset: (125972, 43)

Shape of Testing Dataset: (22543, 43)






DATA PREPROCESSING:

Data preprocessing is a crucial step in machine learning that involves transforming raw data into a suitable format for building predictive models. The main goal of data preprocessing is to ensure that the data is clean, consistent, and in the right format to be used by machine learning algorithms.

The following has been performed in order:

Checking for Null values

Checking for Duplicate Rows

Column Names for Training and Test Data Set

Identify Categorical Features

Add six missing Categories from Train Set to Test Set




We've used the attacks, which are classified into the following four categories:

Denial-of-service (DoS): An attacker tries to make a machine or network resource unavailable to its intended users by overwhelming it with a flood of traffic or by sending malformed packets that cause the resource to crash or become unavailable.

Probe: An attacker sends packets to gather information about a target network or system. This can involve port scanning, fingerprinting, or other techniques that can reveal vulnerabilities or provide information that can be used in a subsequent attack.

Remote-to-local (R2L): An attacker attempts to gain unauthorized access to a target system from a remote location. This can involve exploiting vulnerabilities in network protocols or applications, or using brute-force techniques to crack passwords.

User-to-root (U2R): An attacker who has already gained access to a user account on a target system tries to elevate their privileges to gain root access and take control of the system.


Project Flow and Logic
Step 1: Data Preprocessing

Load the NSL-KDD dataset (train & test)
Convert categorical features into numerical values using One-Hot Encoding
Normalize features to prevent bias from larger values
Step 2: Feature Selection

Use ANOVA F-test to determine important features
Apply Recursive Feature Elimination (RFE) to refine the feature set
Step 3: Model Building

Train a Decision Tree Classifier using the selected features
Split the dataset into train and test sets
Step 4: Model Evaluation & Validation

Evaluate the model using:
Accuracy Score
Recall & F1-score
Confusion Matrix
10-fold Cross-validation




Below is an in-depth explanation of my Intrusion Detection System (IDS) implementation.

Step 1: Importing Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
Logic
NumPy & Pandas: Handling numerical data and datasets.
Matplotlib: Visualizing data.
Scikit-learn modules:
LabelEncoder, OneHotEncoder: Encoding categorical variables.
preprocessing: Feature scaling.
RFE (Recursive Feature Elimination): Selecting important features.
DecisionTreeClassifier: Model training.
train_test_split: Splitting data for training & testing.
Warnings disabled for clean output.
Step 2: Loading the NSL-KDD Dataset

dataset_train = pd.read_csv('NSL_KDD_Train.csv')
dataset_test = pd.read_csv('NSL_KDD_Test.csv')

print("Shape of Training Dataset:", dataset_train.shape)
print("Shape of Testing Dataset:", dataset_test.shape)
Logic
Loads NSL-KDD dataset.
Displays dataset shape.
Step 3: Assigning Column Names


col_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
             "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
             "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
             "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
             "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
             "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
             "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
             
dataset_train = pd.read_csv("NSL_KDD_Train.csv", header=None, names=col_names)
dataset_test = pd.read_csv("NSL_KDD_Test.csv", header=None, names=col_names)
Logic
Defines column names manually.
Re-loads the dataset using assigned column names.
Step 4: Exploring Dataset (Data Distribution)

print("Label distribution Training set:")
print(dataset_train['label'].value_counts())

print("Label distribution Test set:")
print(dataset_test['label'].value_counts())
Logic
Displays the count of attack labels in both train & test sets.
Step 5: Encoding Categorical Features

# Categorical columns: protocol_type (column 2), service (column 3), flag (column 4).
categorical_cols = ['protocol_type', 'service', 'flag']

for col in categorical_cols:
    dataset_train[col] = dataset_train[col].astype('category').cat.codes
    dataset_test[col] = dataset_test[col].astype('category').cat.codes
Logic
Converts categorical columns (protocol_type, service, flag) into numerical values.
Uses category encoding.
Step 6: Feature Selection (ANOVA & RFE)

X = dataset_train.drop(['label'], axis=1)
y = dataset_train['label']

X_test = dataset_test.drop(['label'], axis=1)
y_test = dataset_test['label']

# Using Recursive Feature Elimination
dt = DecisionTreeClassifier()
rfe = RFE(estimator=dt, n_features_to_select=10)
X_rfe = rfe.fit_transform(X, y)
X_test_rfe = rfe.transform(X_test)
Logic
Splits features (X) and labels (y).
Uses Recursive Feature Elimination (RFE) to select top 10 features.
Step 7: Training the Decision Tree Model

X_train, X_val, y_train, y_val = train_test_split(X_rfe, y, test_size=0.2, random_state=42)

dt.fit(X_train, y_train)
y_pred = dt.predict(X_val)
Logic
Splits data into 80% training & 20% validation.
Trains a Decision Tree Classifier.
Step 8: Model Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Accuracy Score:", accuracy_score(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
Logic
Evaluates the model using:
Accuracy
Classification Report (Precision, Recall, F1-score)
Confusion Matrix
Step 9: 10-Fold Cross-Validation

from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt, X_rfe, y, cv=10)
print("Cross-validation scores:", scores)
print("Mean Accuracy:", scores.mean())
Logic
Performs 10-fold cross-validation to check model consistency.










Key Takeaways
✅ Feature Selection: Used ANOVA & RFE to reduce dimensionality.
✅ Model: Decision Tree classifier.
✅ Validation: Used cross-validation & confusion matrix for performance checks.










KNN

The KNN classifier is a simple algorithm that classifies a data point based on the class of its k-nearest neighbors. In other words, it identifies the k training examples in the dataset that are closest to the new data point and predicts the class of the new data point based on the most frequent class among those k neighbors.

Hyperparameters: The hyperparameters of the KNN classifier are k (the number of neighbors to consider) and the distance metric used to calculate the distances between data points (such as Euclidean distance, Manhattan distance, or Minkowski distance).





RANDOM FOREST:

• Random Forest algorithm is a popular machine learning algorithm that is widely used for classification and regression tasks.


• The Y_DoS, Y_Probe, Y_R2L, and Y_U2R variables are converted to integers using the astype(int) method, as the Random ForestClassifier algorithm requires integer labels.







NAIVE BAYES':

• Naive Bayes is a classification algorithm that makes predictions by calculating the probability of each class given a set of input features.

• The main advantage of Naive Bayes is its simplicity and efficiency.

• Four instances of the classifier are created, one for each type of attack.







HAVE A LOOK AT MY PROJECT REPORT IN THE REPOSITORY  which contains output images.

DEPLOYMENT DETAILS:Deployment on Hugging Face Spaces
1️⃣ Create a Hugging Face Space
Go to Hugging Face Spaces
Click "Create new Space"
Choose "Gradio" as the Space SDK
Set visibility to "Public" (or Private if needed)
2️⃣ Upload Files
Upload app.py
Upload network_intrusion_model.pkl (your saved model)
Upload requirements.txt
3️⃣ Run the App
Hugging Face will automatically install dependencies
Your Gradio app will be hosted with a public link





THE DEPLOYED LINK IS:
https://b7f894f91873dc1ab9.gradio.live/

https://huggingface.co/spaces/rohanjain1648/network-intrusion-detection



Gradio helps create a simple web UI for ML models.


SOME INPUT TO CHECK:
0.2, 0.1, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0,0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5
will render 0 




5.1, 3.2, 2.9, 7.5, 4.8, 1.2, 3.0, 2.8, 4.5, 6.1, 2.4, 5.7, 4.9, 6.2, 3.5, 5.8, 2.7, 4.6, 3.1, 5.3, 6.7, 3.9, 7.1, 5.2, 3.4, 6.8, 4.1, 5.6, 2.5, 7.3, 4.4, 3.8, 6.0, 5.9, 7.2, 3.6, 4.2, 5.5, 3.7, 6.5, 4.0, 2.6, 7.0, 5.4, 4.3, 3.3, 6.9, 2.3, 4.7, 3.0, 6.4, 5.0, 7.4, 2.1, 4.9, 3.2, 6.3, 5.1, 7.6, 2.2, 4.8, 3.1, 6.2, 5.0, 7.5, 2.0, 4.7, 3.0, 6.1, 4.9, 7.3, 1.9, 4.6, 2.9, 6.0, 4.8, 7.2, 1.8, 4.5, 2.8, 5.9, 4.7, 7.1, 1.7, 4.4, 2.7, 5.8, 4.6, 7.0, 1.6, 4.3, 2.6, 5.7, 4.5, 6.9, 1.5, 4.2, 2.5, 5.6, 4.4, 6.8, 1.4, 4.1, 2.4, 5.5, 4.3, 6.7, 1.3, 0.0, 0.0, 0.0, 0.0,5.6, 4.4, 6.8, 1.4, 4.1, 2.4, 5.5, 4.3, 6.7, 1.3


will predict 1.







