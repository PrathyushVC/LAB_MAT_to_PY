import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score


def wilcoxon_rank_sum(X, y):
    """
    Perform the Wilcoxon rank-sum test on each feature in dataset.

    Parameters:
    X (numpy.ndarray): A 2D numpy array where each column represents a feature and each row represents an observation.
    y (numpy.ndarray): A 1D numpy array representing the target variable associated with each observation in X.

    Returns:
    numpy.ndarray: An array of p-values indicating the significance of each feature in distinguishing between the classes in y.
    """
    from scipy.stats import ranksums
    p_values = np.array([ranksums(X[:,i], y).pvalue for i in range(X.shape[1])])
    return p_values



def nFoldCV_withFS(data_set, data_labels, params):
    # Set default parameters
    params.setdefault('classifier', 'LDA')
    params.setdefault('classifieroptions', {})
    params.setdefault('fsname', 'wilcoxon')
    params.setdefault('shuffle', 1)
    params.setdefault('n', 3)
    params.setdefault('nIter', 25)
    params.setdefault('num_top_feats', 1)
    params.setdefault('feature_idxs', np.arange(data_set.shape[1]))
    params.setdefault('threshmeth', 'euclidean')
    params.setdefault('classbalancetestfold', 'none')

    assert data_set.shape[0] == len(data_labels), "Mismatch in number of samples and labels"

    # Extract feature indices
    data_set = data_set[:, params['feature_idxs']]
    
    # Initialize statistics
    stats = []
    if params['n'] == data_set.shape[0]:
        print(f'\nExecuting leave-one-out cross-validation using {params["classifier"]} on {len(data_labels)} observations')
    else:
        print(f'\nExecuting {params["nIter"]} run(s) of {params["n"]}-fold cross-validation using {params["classifier"]} on {len(data_labels)} observations')

    if 'osname' in params and params['osname']:
        print(f'\t Oversampling of training data to be performed using {params["osname"]} method')

    if 'remouts' in params and params['remouts']:
        print(f'\t Training data will have outliers removed using {params["remouts"]} method')
        # print('Not recommended. Classifier will probably fail due to NaNs in data')


    if 'patient_ids' in params and params['patient_ids']:
        pts, idxs = np.unique(params['patient_ids'], return_index=True, axis=0)

    for j in range(params['nIter']):
        print(f'Iteration: {j + 1}')
        
        skf = StratifiedKFold(n_splits=params['n'], shuffle=params['shuffle'] == 1)
        
        for i, (train_index, test_index) in enumerate(skf.split(data_set, data_labels)):
            print(f'Fold # {i + 1}')
            training_set, testing_set = data_set[train_index], data_set[test_index]
            training_labels, testing_labels = data_labels[train_index], data_labels[test_index]

            # Feature selection
            if params['num_top_feats'] == training_set.shape[1]:
                selected_features = np.arange(params['num_top_feats'])
            else:
                if params['fsname'] == 'mrmr':
                    # Use mutual information for MRMR equivalent
                    selector = SelectKBest(mutual_info_classif, k=params['num_top_feats']).fit(training_set, training_labels)
                    selected_features = selector.get_support(indices=True)
                elif params['fsname'] == 'wilcoxon':
                    selected_features = wilcoxon_rank_sum(training_set, training_labels)[:params['num_top_feats']]
                else:
                    # Add other feature selection methods if needed
                    selector = SelectKBest(f_classif, k=params['num_top_feats']).fit(training_set, training_labels)
                    selected_features = selector.get_support(indices=True)
            
            training_set = training_set[:, selected_features]
            testing_set = testing_set[:, selected_features]
            
            # Classifier selection
            if params['classifier'] == 'LDA':
                model = LogisticRegression()
            elif params['classifier'] == 'SVM':
                model = SVC(probability=True)
            elif params['classifier'] == 'RANDOMFOREST':
                model = RandomForestClassifier()
            else:
                model = LogisticRegression()  # Default to Logistic Regression
            
            # Fit the model
            model.fit(training_set, training_labels)
            predictions = model.predict_proba(testing_set)[:, 1]
            
            # Compute ROC AUC
            fpr, tpr, _ = roc_curve(testing_labels, predictions)
            roc_auc = roc_auc_score(testing_labels, predictions)
            stats.append({
                'fold': i + 1,
                'iteration': j + 1,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'selected_features': selected_features
            })
    
    return stats
    

