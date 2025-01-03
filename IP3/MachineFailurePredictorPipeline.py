import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

'''The functions below are responsible for fitting the models and 
identifying the best parameters for each. They use GridSearchCV to
perform the 5-fold CV and get the evaluation of the training set.
The prediction of the test set is in the main function using the
model created in the helper functions'''

def k_nearest(X_train, y_train, scoring_function):
    k_args = {'n_neighbors':[3,5,7,11,13,100], 'p':[1,2,3,4,5]}
    k_classifier = KNeighborsClassifier()
    k_classifier.fit(X_train, y_train)
    k_gridsearch = GridSearchCV(k_classifier, k_args, scoring = scoring_function,
                                 return_train_score=True, refit='mcc')
    k_gridsearch.fit(X_train, y_train)
    k_pred = k_gridsearch.predict(X_train)
    return k_gridsearch, k_pred

def decision_tree(X_train, y_train, scoring_function):
    dt_args = {'ccp_alpha':[0.0,0.015,0.03,0.05], 'criterion':['gini','entropy','log_loss']}
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    dt_gridsearch = GridSearchCV(dt_classifier, dt_args, return_train_score=True,
                                 scoring = scoring_function, refit='mcc')
    dt_gridsearch.fit(X_train, y_train)
    dt_pred = dt_gridsearch.predict(X_train)
    return dt_gridsearch, dt_pred

def ANN(X_train, y_train, scoring_function):
    ann_args = {'hidden_layer_sizes':[(50,48),(100,98),(20,18)],
                'activation':['identity','logistic','tanh','relu']}
    ann_classifier = MLPClassifier(early_stopping = True)
    ann_classifier.fit(X_train, y_train)
    ann_gridsearch = GridSearchCV(ann_classifier, ann_args, return_train_score=True,
                                  scoring = scoring_function, refit='mcc')
    ann_gridsearch.fit(X_train, y_train)
    ann_pred = ann_gridsearch.predict(X_train)
    return ann_gridsearch, ann_pred

def log_reg(X_train, y_train, scoring_function):
    '''I don't need to use GridSearch here since there is no tuning
    involved, but I used it for consistency since it uses a 5-fold cv
    by default and I can get the MCC this way'''
    log_args = {'C':[1]}
    log_classifier = LogisticRegression()
    log_classifier.fit(X_train, y_train)
    log_gridsearch = GridSearchCV(log_classifier, log_args, return_train_score=True,
                                  scoring=scoring_function, refit='mcc')
    log_gridsearch.fit(X_train, y_train)
    log_pred = log_gridsearch.predict(X_train)
    return log_gridsearch, log_pred

def SVM(X_train, y_train, scoring_function):
    svm_args = {'C':[.5,1,1.5], 'kernel':['linear', 'poly', 'rbf','sigmoid']}
    svm_classifier = SVC()
    svm_classifier.fit(X_train, y_train)
    svm_gridsearch = GridSearchCV(svm_classifier, svm_args, return_train_score=True,
                                  scoring=scoring_function, refit='mcc')
    svm_gridsearch.fit(X_train, y_train)
    svm_pred = svm_gridsearch.predict(X_train)
    return svm_gridsearch, svm_pred


def main():
    ##STEP 3
    # Read in the data to a dataframe
    df = pd.read_csv('ai4i2020.csv', usecols=['Type', 'Air temperature [K]',
                                              'Process temperature [K]', 'Rotational speed [rpm]',
                                              'Torque [Nm]', 'Tool wear [min]', 'Machine failure'])

    print(f'STEP 3 Full Dataset:\n{df}\n')

    ##STEP 4
    # Factorize the 'Type' column so turn text data into numerical data
    labels, uniques = pd.factorize(df['Type'])
    df['Type'] = labels
    
    #Normalize the columns
    df = (df-df.min())/(df.max()-df.min())
    print(f'STEP 4 Processed Dataset:\n{df}\n')
    ##STEP 5
    #Random under sample
    rus = RandomUnderSampler(random_state=42)
    x_rus, y_rus = rus.fit_resample(df, df['Machine failure'])
    print(f'STEP 5 Undersampled Dataset:\n{x_rus}\n')

    ##STEP 6
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(x_rus[['Type', 'Air temperature [K]',
                                              'Process temperature [K]', 'Rotational speed [rpm]',
                                              'Torque [Nm]', 'Tool wear [min]']],
                                              x_rus['Machine failure'], test_size = .3)
    
    #Create a scoring function that uses the MCC as the metric for GridSearch to select
    # the best parameters
    scoring_function = {'mcc':make_scorer(matthews_corrcoef)}


    # Create all the classifiers, perform finetuning and fit models
    k_gridsearch, k_pred = k_nearest(X_train, y_train, scoring_function)
    dt_gridsearch, dt_pred = decision_tree(X_train, y_train, scoring_function)
    svm_gridsearch, svm_pred = SVM(X_train, y_train, scoring_function)
    ann_gridsearch, ann_pred = ANN(X_train, y_train, scoring_function)
    log_gridsearch, log_pred = log_reg(X_train, y_train, scoring_function)
    
    print(ann_gridsearch.best_estimator_)
    
    #Generate Table
    table_entries = [['k_nearest', k_gridsearch.best_params_, matthews_corrcoef(y_train, k_pred)],
                     ['Decision Tree', dt_gridsearch.best_params_, matthews_corrcoef(y_train, dt_pred)],
                     ['SVM', svm_gridsearch.best_params_, matthews_corrcoef(y_train, svm_pred)],
                     ['Artificial NN', ann_gridsearch.best_params_, matthews_corrcoef(y_train, ann_pred)],
                     ['Log Regression', 'NA', matthews_corrcoef(y_train, log_pred)]]
    table = pd.DataFrame(table_entries, columns=['Model','Tuned Hyperparameters','MCC Training Set Score'])
    print(f'STEP 6:\n{table}\n')

    #Create all the predictions on test set
    k_prediction = k_gridsearch.predict(X_test)
    dt_prediction = dt_gridsearch.predict(X_test)
    svm_prediction = svm_gridsearch.predict(X_test)
    ann_prediction = ann_gridsearch.predict(X_test)
    log_prediction = log_gridsearch.predict(X_test)
    
    #Count the number of correct predictions
    k_correct = np.where(k_prediction == y_test, 1, 0).sum()
    dt_correct = np.where(dt_prediction == y_test, 1, 0).sum()
    svm_correct = np.where(svm_prediction == y_test, 1, 0).sum()
    ann_correct = np.where(ann_prediction == y_test, 1, 0).sum()
    log_correct = np.where(log_prediction == y_test,1, 0).sum()

    table_entries = [['k_nearest', k_gridsearch.best_params_, matthews_corrcoef(y_test, k_prediction), k_correct],
                     ['Decision Tree', dt_gridsearch.best_params_, matthews_corrcoef(y_test, dt_prediction), dt_correct],
                     ['SVM', svm_gridsearch.best_params_, matthews_corrcoef(y_test, svm_prediction), svm_correct],
                     ['Artificial NN', ann_gridsearch.best_params_, matthews_corrcoef(y_test, ann_prediction), ann_correct],
                     ['Log Regression', 'NA', matthews_corrcoef(y_test, log_prediction), log_correct]]

    table = pd.DataFrame(table_entries, columns=['Model','Tuned Hyperparameters','MCC Test Set Score', 'Number Correct'])
    print(f'STEP 7:\n{table}\n')
    

if __name__ == '__main__':
    main()
