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
# for i in range(N):
#     # Randomly shuffle the indices for the dataset
#     indices = np.random.permutation(len(X))
    
#     # Select a random subset size (n <= n_max)
#     # n = np.random.randint(1, n_max + 1)
#     n=20
    
#     # Split indices into training and test subsets
#     training_indices = indices[:n]
#     test_indices = indices[n:]
    
#     # Create training and test subsets
#     X_train = X[training_indices]
#     X_test = X[test_indices]
#     y_train = y[training_indices]
#     y_test = y[test_indices]
    
#     training_sets.append(X_train)
#     test_sets.append(X_test)
#     training_labels.append(y_train)
#     test_labels.append(y_test)

#  creating the training and test sets differently to ensure the classes and samples are balanced for later
for i in range(N):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, stratify=y  
    )
    n = 20 # always choose 20 now because it was causing issues if it selected too few samples for training
    
    training_sets.append(X_train[:n]) 
    test_sets.append(X_test)
    training_labels.append(y_train[:n])
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
    resub_accuracy_list = []
    resub_bias_list = []
    resub_variance_list = []

    loo_accuracy_list = []
    loo_bias_list = []
    loo_variance_list = []

    five_fold_accuracy_list = []
    five_fold_bias_list = []
    five_fold_variance_list = []

    bootstrap_accuracy_list = []
    bootstrap_bias_list = []
    bootstrap_variance_list = []

    bolstered_accuracy_list = []
    bolstered_bias_list = []
    bolstered_variance_list = []

    # Loop over N simulations to evaluate on each simulation
    for i in range(N):
        # Get the current simulation's training and test set
        X_train = training_sets[i]
        X_test = test_sets[i]
        y_train = training_labels[i]
        y_test = test_labels[i]

        # Train the model on the training set
        clf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Resubstitution 
        resub_accuracy = resubstitution(clf, X_train, y_train)  # Resubstitution is evaluated on the training set
        resub_bias, resub_variance = calculate_bias_variance(y_train, clf.predict(X_train)) 
        resub_accuracy_list.append(resub_accuracy)
        resub_bias_list.append(resub_bias)
        resub_variance_list.append(resub_variance)

        ## Leave-One-Out 
        # loo_accuracy = loo_cv(clf, X_train, y_train) 
        # loo_bias, loo_variance = calculate_bias_variance(y_test, y_pred)
        # loo_accuracy_list.append(loo_accuracy)
        # loo_bias_list.append(loo_bias)
        # loo_variance_list.append(loo_variance)

        ## 5-Fold 
        # five_fold_accuracy = five_fold_cv(clf, X_train, y_train)  
        # five_fold_bias, five_fold_variance = calculate_bias_variance(y_test, y_pred)
        # five_fold_accuracy_list.append(five_fold_accuracy)
        # five_fold_bias_list.append(five_fold_bias)
        # five_fold_variance_list.append(five_fold_variance)

        ## Bootstrap .632 
        # bootstrap_accuracy = bootstrap_632(clf, X_train, y_train)  
        # bootstrap_bias, bootstrap_variance = calculate_bias_variance(y_test, y_pred)
        # bootstrap_accuracy_list.append(bootstrap_accuracy)
        # bootstrap_bias_list.append(bootstrap_bias)
        # bootstrap_variance_list.append(bootstrap_variance)

        # # Bolstered Resubstitution 
        # bolstered_accuracy = bolstered_resubstitution(clf, X_train, y_train)  
        # bolstered_bias, bolstered_variance = calculate_bias_variance(y_test, y_pred)
        # bolstered_accuracy_list.append(bolstered_accuracy)
        # bolstered_bias_list.append(bolstered_bias)
        # bolstered_variance_list.append(bolstered_variance)

    avg_resub_accuracy = np.mean(resub_accuracy_list)
    avg_resub_bias = np.mean(resub_bias_list)
    avg_resub_variance = np.mean(resub_variance_list)

    # avg_loo_accuracy = np.mean(loo_accuracy_list)
    # avg_loo_bias = np.mean(loo_bias_list)
    # avg_loo_variance = np.mean(loo_variance_list)

    # avg_five_fold_accuracy = np.mean(five_fold_accuracy_list)
    # avg_five_fold_bias = np.mean(five_fold_bias_list)
    # avg_five_fold_variance = np.mean(five_fold_variance_list)

    # avg_bootstrap_accuracy = np.mean(bootstrap_accuracy_list)
    # avg_bootstrap_bias = np.mean(bootstrap_bias_list)
    # avg_bootstrap_variance = np.mean(bootstrap_variance_list)

    # avg_bolstered_accuracy = np.mean(bolstered_accuracy_list)
    # avg_bolstered_bias = np.mean(bolstered_bias_list)
    # avg_bolstered_variance = np.mean(bolstered_variance_list)

    results[clf_name] = {
        "Resubstitution": {
            "accuracy": avg_resub_accuracy,
            "error": 1 - avg_resub_accuracy,
            "bias": avg_resub_bias,
            "variance": avg_resub_variance
        },
        # "LOO-CV": {
        #     "accuracy": avg_loo_accuracy,
        #     "error": 1 - avg_loo_accuracy,
        #     "bias": avg_loo_bias,
        #     "variance": avg_loo_variance
        # },
        # "5-Fold CV": {
        #     "accuracy": avg_five_fold_accuracy,
        #     "error": 1 - avg_five_fold_accuracy,
        #     "bias": avg_five_fold_bias,
        #     "variance": avg_five_fold_variance
        # },
        # "Bootstrap 632": {
        #     "accuracy": avg_bootstrap_accuracy,
        #     "error": 1 - avg_bootstrap_accuracy,
        #     "bias": avg_bootstrap_bias,
        #     "variance": avg_bootstrap_variance
        # },
        # "Bolstered Resubstitution": {
        #     "accuracy": avg_bolstered_accuracy,
        #     "error": 1 - avg_bolstered_accuracy,
        #     "bias": avg_bolstered_bias,
        #     "variance": avg_bolstered_variance
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