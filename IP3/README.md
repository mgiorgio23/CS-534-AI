Matthew Giorgio
7/23/23
CS 534
IP3

STEP 3:
    CSV was read into a dataframe using only the columns of interest:
        - Type, Air Temp, Process Temp, Rotation speed, Torque, Tool wear, Machine failure

STEP 4:
    The 'Type' column was transformed so that Low, Med, and High were represented by 1,2,3
    The dataframe was then normalized so that each value falls between 0 and 
    **This is the first dataframe shown in the console output

STEP 5:
    Used RandomUnderSampler to isolate the 339 instances of machine failure, 
        and an equally sized sample of non-failure data points
    Used imblearn library for this step
    **This is the second dataframe shown in the console output

STEP 6:
    Used train_test_split to split the preprocessed data into training and testing sets (70% train, 30% test)
    Used GridSearchCV to finetune the hyperparameters:
        - GridSearchCV performs a 5-fold CV by default and returns the best parameters determined by MCC score
        - Used make_scorer to evaluate each of the folds with MCC score
    After fitting the GridSearchCV object, GridSearchCV.predict() uses the parameters that had just been identified,
        so I run the prediction on the training set and collect the MCC 
    ** The results of this populate the first table in the console output

STEP 7:
    Now the model is used to make a prediction on the test set of data
    ** The results of this populate the second table in the console output
        I added a fourth column that counts the number of correct predictions,
        which is out of 204 data points (30% of the total data)

The results were also significantly worse and took longer to generate when the data was not normalized, which 
    highlights the importance of data preprocessing.

**Overall, the decision tree appears to perform the best out of all the models. I wasn't expecting this,
but perhaps due to the limited data, it is best suited for this type of analysis. I expected the ANN to perform
the worst due to the data constraints.