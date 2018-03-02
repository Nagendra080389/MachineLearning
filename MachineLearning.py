# Load required libraries and datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


def plot_learning_curve(
        estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve. Parameters ---------- estimator :
    object type that implements the "fit" and "predict" methods An object of that type which is cloned for each validation. title :
    string Title for the chart. X : array-like, shape (n_samples, n_features) Training vector, where n_samples is the number of samples and n_features
    is the number of features. y : array-like, shape (n_samples) or (n_samples, n_features), optional Target relative to X for classification or
    regression; None for unsupervised learning. ylim : tuple, shape (ymin, ymax), optional Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional If an integer is passed, it is the number of folds (defaults to 3).
    Specific cross-validation objects can be passed, see sklearn.cross_validation module for the list of possible objects n_jobs :
    integer, optional Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt


if __name__ == '__main__':

    path = "C:\ML\AirBnB\\"
    train = pd.read_csv(path + "train.csv")
    test = pd.read_csv(path + "test.csv")

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
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
    categorical = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'neighbourhood']
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
    # sample_leaf_options = [100]

    # regr = RandomForestRegressor(n_estimators = 500, oob_score = False, n_jobs = -1,max_features = 0.87)

    # regr = GradientBoostingRegressor(n_estimators=1000,max_depth=5,min_samples_split =2,learning_rate=0.05,loss='ls')
    regr = GradientBoostingRegressor(n_estimators= 1410,max_depth= 4,learning_rate=0.1,min_samples_leaf=60,max_features=1.0)
    # regr = RandomForestRegressor(bootstrap=True, criterion='mse',
    #        max_features=0.83, max_leaf_nodes=None,
    #        min_impurity_decrease=0.0, min_impurity_split=None,
    #        min_samples_leaf=10, min_samples_split=3,
    #        min_weight_fraction_leaf=0.0, n_estimators=1500, n_jobs=-1,
    #        oob_score=False, random_state=None, verbose=0, warm_start=False)
    # Random search of par  ameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    # rf_random = RandomizedSearchCV(estimator = regr, param_distributions = random_grid, n_iter = 5, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    for train_index, test_index in cv_groups.split(train_x):
        # Train the model using the training sets
        regr.fit(train_x[train_index], train_y[train_index])

        # Make predictions using the testing set
        pred_rf = regr.predict(train_x[test_index])

        # Calculate RMSE for current cross-validation split
        rmse = str(np.sqrt(np.mean((train_y[test_index] - pred_rf) ** 2)))
        print("RMSE for current split: " + rmse)
        # print "AUC - ROC : ", roc_auc_score(train_y[train_index], regr.oob_prediction_)

        # Create submission file
    title = "Learning Curves (Gradient Boosted Regression Trees)"
    regr.fit(train_x, train_y)
    # Re-plotting Learning cruves.
    # plot_learning_curve(regr, title, train_x, train_y, cv=3, n_jobs=4)
    # plt.show()
    final_prediction = regr.predict(test_x)

    submission = pd.DataFrame(np.column_stack([test.id, final_prediction]), columns=['id', 'log_price'])
    submission.to_csv("sample_submission" + ".csv", index=False)

    # regr.fit(train_x, train_y)

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



