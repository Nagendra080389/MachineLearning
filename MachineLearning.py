# Load required libraries and datasets

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':

    path = "C:\ML\AirBnB\\"
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # One-hot-encode categorical variables
    train['dataset'] = "train"
    test['dataset'] = "test"
    data = pd.concat([train, test], axis=0)
    categorical = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city','neighbourhood']
    data = pd.get_dummies(data, columns=categorical)

    # Select only numeric data and impute missing values as 0
    numerics = ['uint8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    train_x = data[data.dataset == "train"] \
        .select_dtypes(include=numerics) \
        .drop("log_price", axis=1) \
        .fillna(0) \
        .values

    test_x = data[data.dataset == "test"] \
        .select_dtypes(include=numerics) \
        .drop("log_price", axis=1) \
        .fillna(0) \
        .values

    train_y = data[data.dataset == "train"].log_price.values

    # Train a Random Forest model with cross-validation

    from sklearn.model_selection import KFold

    cv_groups = KFold(n_splits=3)

    sample_leaf_options = [5]
    #sample_leaf_options = [100]

    #regr = RandomForestRegressor(n_estimators = 500, oob_score = False, n_jobs = -1,max_features = 0.87)

    regr = RandomForestRegressor()
    # Random search of par  ameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    for train_index, test_index in cv_groups.split(train_x):
            # Train the model using the training sets
        rf_random.fit(train_x[train_index], train_y[train_index])


        # Make predictions using the testing set
        pred_rf = rf_random.predict(train_x[test_index])

        # Calculate RMSE for current cross-validation split
        rmse = str(np.sqrt(np.mean((train_y[test_index] - pred_rf) ** 2)))
        print("RMSE for current split: " + rmse)
        # print "AUC - ROC : ", roc_auc_score(train_y[train_index], regr.oob_prediction_)

        # Create submission file
    rf_random.fit(train_x, train_y)
    final_prediction = rf_random.predict(test_x)

    submission = pd.DataFrame(np.column_stack([test.id, final_prediction]), columns=['id', 'log_price'])
    submission.to_csv("sample_submission"+".csv", index=False)

    #regr.fit(train_x, train_y)

    # print('----------')
    # print(regr.best_estimator_)
    # print(regr.best_params_)
    # print('----------')
    # pred_rf = regr.predict(train_x)
    #
    # rmse = str(np.sqrt(np.mean((train_y - pred_rf) ** 2)))
    # print("RMSE for current split: " + rmse)
    #
    # final_prediction = regr.predict(test_x)
    #
    # submission = pd.DataFrame(np.column_stack([test.id, final_prediction]), columns=['id', 'log_price'])
    # submission.to_csv("sample_submission" + ".csv", index=False)





