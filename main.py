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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
    n = np.random.randint(20, 50)
    
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
    return (1-accuracy)

# Leave-One-Out Cross-Validation (LOO-CV)
def loo_cv(model, X, y):
    loo = LeaveOneOut()
    print("")

# 5-Fold Cross-Validation
def five_fold_cv(model, X, y):
    print("")

# .632 Bootstrap
def bootstrap_632(model, X, y, n_iterations=2):
    err = 0.0
    n_samples = len(X)
    classes, class_counts = np.unique(y, return_counts=True)
    
    if len(classes) < 2:
        raise ValueError("Dataset must have at least two classes.")

    for _ in range(n_iterations):
        while True:
            # Perform stratified bootstrap sampling
            X_boot = []
            y_boot = []
            
            for cls in classes:
                cls_indices = np.where(y == cls)[0]
                if len(cls_indices) < 2:  # Ensure at least 2 samples per class
                    raise ValueError(f"Class {cls} has fewer than 2 samples.")
                sampled_indices = np.random.choice(cls_indices, size=len(cls_indices), replace=True)
                X_boot.append(X[sampled_indices])
                y_boot.append(y[sampled_indices])
            
            X_boot = np.vstack(X_boot)
            y_boot = np.hstack(y_boot)
            
            # Ensure at least two unique classes are present
            if len(np.unique(y_boot)) == len(classes):
                break
        
        # Fit the model on the bootstrap sample
        model.fit(X_boot, y_boot)

        # Predict on out-of-bag samples
        oob_indices = np.setdiff1d(np.arange(n_samples), sampled_indices)
        if len(oob_indices) > 0:  # Ensure OOB samples are available
            y_pred_oob = model.predict(X[oob_indices])

            # Calculate the OOB error
            err += 1 - accuracy_score(y[oob_indices], y_pred_oob)

    # Calculate the .632 Bootstrap error estimation
    err /= n_iterations
    err_632 = 0.632 * err + 0.368 * (1 - accuracy_score(y, model.predict(X)))

    return err_632


def bolstered_resubstitution(model, X, y):
    model.fit(X, y)

    y_pred = model.predict(X)

    error_count = 0
    for i in range(len(X)):
        if y_pred[i] != y[i]:
            error_count += 1

    # Bolstering: Add a small positive value to the error count
    error_count += 0.5

    return error_count / len(X)

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
    resub_err_list = []
    resub_bias_list = []
    resub_variance_list = []

    loo_err_list = []
    loo_bias_list = []
    loo_variance_list = []

    five_fold_err_list = []
    five_fold_bias_list = []
    five_fold_variance_list = []

    bootstrap_err_list = []
    bootstrap_bias_list = []
    bootstrap_variance_list = []

    bolstered_err_list = []
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
        resub_err = resubstitution(clf, X_train, y_train)  # Resubstitution is evaluated on the training set
        resub_bias, resub_variance = calculate_bias_variance(y_train, clf.predict(X_train)) 
        resub_err_list.append(resub_err)
        resub_bias_list.append(resub_bias)
        resub_variance_list.append(resub_variance)

        ## Leave-One-Out 
        # loo_err = loo_cv(clf, X_train, y_train) 
        # loo_bias, loo_variance = calculate_bias_variance(y_test, y_pred)
        # loo_err_list.append(loo_err)
        # loo_bias_list.append(loo_bias)
        # loo_variance_list.append(loo_variance)

        ## 5-Fold 
        # five_fold_err = five_fold_cv(clf, X_train, y_train)  
        # five_fold_bias, five_fold_variance = calculate_bias_variance(y_test, y_pred)
        # five_fold_err_list.append(five_fold_err)
        # five_fold_bias_list.append(five_fold_bias)
        # five_fold_variance_list.append(five_fold_variance)

        # .632 Bootstrap 
        bootstrap_err = bootstrap_632(clf, X_train, y_train)  
        bootstrap_bias, bootstrap_variance = calculate_bias_variance(y_test, y_pred)
        bootstrap_err_list.append(bootstrap_err)
        bootstrap_bias_list.append(bootstrap_bias)
        bootstrap_variance_list.append(bootstrap_variance)

        # Bolstered Resubstitution 
        bolstered_err = bolstered_resubstitution(clf, X_train, y_train)  
        bolstered_bias, bolstered_variance = calculate_bias_variance(y_test, y_pred)
        bolstered_err_list.append(bolstered_err)
        bolstered_bias_list.append(bolstered_bias)
        bolstered_variance_list.append(bolstered_variance)

    avg_resub_err = np.mean(resub_err_list)
    avg_resub_bias = np.mean(resub_bias_list)
    avg_resub_variance = np.mean(resub_variance_list)

    # avg_loo_err = np.mean(loo_err_list)
    # avg_loo_bias = np.mean(loo_bias_list)
    # avg_loo_variance = np.mean(loo_variance_list)

    # avg_five_fold_err = np.mean(five_fold_err_list)
    # avg_five_fold_bias = np.mean(five_fold_bias_list)
    # avg_five_fold_variance = np.mean(five_fold_variance_list)

    avg_bootstrap_err = np.mean(bootstrap_err_list)
    avg_bootstrap_bias = np.mean(bootstrap_bias_list)
    avg_bootstrap_variance = np.mean(bootstrap_variance_list)

    avg_bolstered_err = np.mean(bolstered_err_list)
    avg_bolstered_bias = np.mean(bolstered_bias_list)
    avg_bolstered_variance = np.mean(bolstered_variance_list)

    results[clf_name] = {
        "Resubstitution": {
            "accuracy": 1 - avg_resub_err,
            "error": avg_resub_err,
            "bias": avg_resub_bias,
            "variance": avg_resub_variance
        },
        # "LOO-CV": {
        #     "accuracy": 1 - avg_loo_err,
        #     "error": avg_loo_err,
        #     "bias": avg_loo_bias,
        #     "variance": avg_loo_variance
        # },
        # "5-Fold CV": {
        #     "accuracy": 1 - avg_five_fold_err,
        #     "error": avg_five_fold_err,
        #     "bias": avg_five_fold_bias,
        #     "variance": avg_five_fold_variance
        # },
        "Bootstrap 632": {
            "accuracy": 1 - avg_bootstrap_err,
            "error": avg_bootstrap_err,
            "bias": avg_bootstrap_bias,
            "variance": avg_bootstrap_variance
        },
        "Bolstered Resubstitution": {
            "accuracy": 1 - avg_bolstered_err,
            "error": avg_bolstered_err,
            "bias": avg_bolstered_bias,
            "variance": avg_bolstered_variance
        }
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