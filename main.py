import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.utils import resample

##################################################
# Load the dataset
##################################################

file_path = 'data/raw_data2020.p'

data = pickle.load(open(file_path, "rb"))

##################################################
# Feature Identification/Extraction
##################################################

SSQs, Nauseas, Oculomotors, Disorientations = [], [], [], []

# Loop through the dataset to collect all of the SSQs and other scores
for key, values in data.items():
    SSQs.append(values.SicknessLevel.SSQ)
    Nauseas.append(values.SicknessLevel.Nausea)
    Oculomotors.append(values.SicknessLevel.Oculomotor)
    Disorientations.append(values.SicknessLevel.Disorientation)

# Now head tracking data
head_position_data = []
for key, values in data.items():
    head_position_data.append({
        "position_x": np.mean(values.Steam.rawHead.local_X) if isinstance(values.Steam.rawHead.local_X, (list, pd.Series, np.ndarray)) else values.Steam.rawHead.local_X,
        "position_y": np.mean(values.Steam.rawHead.local_Y) if isinstance(values.Steam.rawHead.local_Y, (list, pd.Series, np.ndarray)) else values.Steam.rawHead.local_Y,
        "position_z": np.mean(values.Steam.rawHead.local_Z) if isinstance(values.Steam.rawHead.local_Z, (list, pd.Series, np.ndarray)) else values.Steam.rawHead.local_Z,
        "rotation_x": np.mean(values.Steam.rawHead.local_EulerAngles_X) if isinstance(values.Steam.rawHead.local_EulerAngles_X, (list, pd.Series, np.ndarray)) else values.Steam.rawHead.local_EulerAngles_X,
        "rotation_y": np.mean(values.Steam.rawHead.local_EulerAngles_Y) if isinstance(values.Steam.rawHead.local_EulerAngles_Y, (list, pd.Series, np.ndarray)) else values.Steam.rawHead.local_EulerAngles_Y,
        "rotation_z": np.mean(values.Steam.rawHead.local_EulerAngles_Z) if isinstance(values.Steam.rawHead.local_EulerAngles_Z, (list, pd.Series, np.ndarray)) else values.Steam.rawHead.local_EulerAngles_Z
    })

# Now biosignal data
biosignals_data = []
for key, values in data.items():
    biosignals_data.append({
        "BVP": np.mean(values.Empatica.BVP.BVP) if isinstance(values.Empatica.BVP.BVP, (list, pd.Series, np.ndarray)) else values.Empatica.BVP.BVP,
        "GSR": np.mean(values.Empatica.GSR.GSR) if isinstance(values.Empatica.GSR.GSR, (list, pd.Series, np.ndarray)) else values.Empatica.GSR.GSR,
        "HR": np.mean(values.Empatica.HR.HR) if isinstance(values.Empatica.HR.HR, (list, pd.Series, np.ndarray)) else values.Empatica.HR.HR,
        "TEM": np.mean(values.Empatica.TEM.TEM) if isinstance(values.Empatica.TEM.TEM, (list, pd.Series, np.ndarray)) else values.Empatica.TEM.TEM
    })

##################################################
# Preprocessing
##################################################

# Putting it into to DataFrames
scores = pd.DataFrame({
    "SSQ": SSQs,
    "Nausea": Nauseas,
    "Oculomotor": Oculomotors,
    "Disorientation": Disorientations
})
head_data = pd.DataFrame(head_position_data)
biosignals_data = pd.DataFrame(biosignals_data)

# Combine all features into a single DataFrame
features_df = pd.concat([scores, head_data, biosignals_data], axis=1)
# print("Combined Feature Dataset:")
# print(features_df.head())

# Handle missing values
features_df = features_df.fillna(features_df.mean())

# Separate features and labels (SSQ scores are probably the best labels for us to use)
labels = features_df["SSQ"]
features = features_df.drop(columns=["SSQ"])

# Normalize 
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

X = np.array(features_normalized)
y = np.array(labels)
# print(f"Feature matrix shape: {X.shape}")
# print(f"Label array shape: {y.shape}")

##################################################
# Small Training Subset Creation
##################################################

    # 0-10: Generally considered no cybersickness or mild symptoms.
    # 10-30: Mild to moderate cybersickness.
    # 30-60: Moderate to severe cybersickness.
    # Above 60: Severe cybersickness.
threshold = 30
y_binary = (labels > threshold).astype(int)  # 0 for no sickness, 1 for sickness


y = np.array(y_binary)

N = 1000  # Number of simulations
n_max = 20  # Maximum subset size

training_sets = []
test_sets = []
training_labels = []
test_labels = []

# Create all of the training and test sets over N sims
for i in range(N):
    # Randomly shuffle the indices for the dataset
    indices = np.random.permutation(len(X))
    
    # Select a random subset size (n <= n_max)
    n = np.random.randint(1, n_max + 1)
    
    # Split indices into training and test subsets
    training_indices = indices[:n]
    test_indices = indices[n:]
    
    # Create training and test subsets
    X_train = X[training_indices]
    X_test = X[test_indices]
    y_train = y[training_indices]
    y_test = y[test_indices]
    
    training_sets.append(X_train)
    test_sets.append(X_test)
    training_labels.append(y_train)
    test_labels.append(y_test)



##################################################
# Classifiers and Cross-Validation
##################################################

# Define the classifiers
classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "3NN": KNeighborsClassifier(n_neighbors=3),
    "SVM": SVC(kernel='linear')
}

# Cross-Validation Methods 
# Resubstitution
def resubstitution(model, X, y): 
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy

# Leave-One-Out Cross-Validation (LOO-CV)
def loo_cv(model, X, y):
    loo = LeaveOneOut()
    print("")

# 5-Fold Cross-Validation
def five_fold_cv(model, X, y):
    print("")

# .632 Bootstrap
def bootstrap_632(model, X, y, n_iterations=1000):
    print("")


# Bolstered Resubstitution
def bolstered_resubstitution(model, X, y, holdout_size=0.1):
   print("")


##################################################
# Performance Metrics and Error Analysis 
##################################################
# Bias: Difference between predicted and actual values.
# Variance: Variability of predictions across subsets.
# True Classification Error: Direct error from test sets.

def calculate_bias_variance(y_true, y_pred):
    bias_squared = np.mean(y_pred - y_true) ** 2
    variance = np.var(y_pred)
    return bias_squared, variance



# Evaluate all classifiers with each of the cross-validation methods
results = {}

for clf_name, clf in classifiers.items():
    resub_accuracy = resubstitution(clf, X, y)
    y_pred_resub = clf.predict(X)
    resub_bias, resub_variance = calculate_bias_variance(y, y_pred_resub)

    # loo_accuracy = loo_cv(clf, X, y)
    # y_pred_loo = clf.predict(X)  
    # loo_bias, loo_variance = calculate_bias_variance(y, y_pred_loo)

    # five_fold_accuracy = five_fold_cv(clf, X, y)
    # y_pred_five_fold = clf.predict(X)
    # five_fold_bias, five_fold_variance = calculate_bias_variance(y, y_pred_five_fold)

    # bootstrap_accuracy = bootstrap_632(clf, X, y)
    # y_pred_bootstrap = clf.predict(X)
    # bootstrap_bias, bootstrap_variance = calculate_bias_variance(y, y_pred_bootstrap)

    # bolstered_accuracy = bolstered_resubstitution(clf, X, y)
    # y_pred_bolstered = clf.predict(X)
    # bolstered_bias, bolstered_variance = calculate_bias_variance(y, y_pred_bolstered)

    results[clf_name] = {
        "Resubstitution": {
            "accuracy": resub_accuracy,
            "error": 1 - resub_accuracy,
            "bias": resub_bias,
            "variance": resub_variance
        },
        # "LOO-CV": {
        #     "accuracy": loo_accuracy,
        #     "error": 1 - loo_accuracy,
        #     "bias": loo_bias,
        #     "variance": loo_variance
        # },
        # "5-Fold CV": {
        #     "accuracy": five_fold_accuracy,
        #     "error": 1 - five_fold_accuracy,
        #     "bias": five_fold_bias,
        #     "variance": five_fold_variance
        # },
        # "Bootstrap 632": {
        #     "accuracy": bootstrap_accuracy,
        #     "error": 1 - bootstrap_accuracy,
        #     "bias": bootstrap_bias,
        #     "variance": bootstrap_variance
        # },
        # "Bolstered Resubstitution": {
        #     "accuracy": bolstered_accuracy,
        #     "error": 1 - bolstered_accuracy,
        #     "bias": bolstered_bias,
        #     "variance": bolstered_variance
        # }
    }

# Print results
for clf_name, metrics in results.items():
    print(f"\n{clf_name} Results:")
    for method, stats in metrics.items():
        print(f"{method}:")
        print(f"  Accuracy: {stats['accuracy']:.4f}")
        print(f"  Error: {stats['error']:.4f}")
        print(f"  Bias: {stats['bias']:.4f}")
        print(f"  Variance: {stats['variance']:.4f}")

##################################################
# Visualization (Plotting)
##################################################
# Includes deviation distributions by beta fitting