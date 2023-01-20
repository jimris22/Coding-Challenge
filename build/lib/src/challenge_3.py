import argparse
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline():
    """
    This function creates the pipeline object that includes the RandomForestRegressor and sets it up for
    hyperparameter tuning :return: pipeline object :rtype: sklearn.pipeline.Pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('reg', RandomForestRegressor())
    ])
    return pipeline


def tune_hyperparameters(pipeline, X, y):
    """
    This function performs hyperparameter tuning on the given pipeline and data
    :param pipeline: pipeline object to tune hyperparameters for
    :type pipeline: sklearn.pipeline.Pipeline
    :param X: input feature variables
    :type X: array-like
    :param y: target variable
    :type y: array-like
    :return: best parameters of the model
    :rtype: dict
    """
    try:
        param_grid = {
            'reg__n_estimators': [50, 100, 200],
            'reg__max_depth': [None, 5, 10],
            'reg__min_samples_split': [2, 5, 10],
            'reg__min_samples_leaf': [1, 2, 4],
            'reg__max_features': ['sqrt', 'log2', None]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        best_params = grid_search.best_params_
        return best_params
    except:
        raise ValueError("An error occured while tuning the hyperparameters")


def train_model(X_train, y_train, best_params):
    """
    This function trains the model with the best hyperparameters
    :param X_train: input feature variables for training
    :type X_train: array-like
    :param y_train: target variable for training
    :type y_train: array-like
    :param best_params: best hyperparameters of the model
    :type best_params: dict
    :return: trained model
    """
    try:
        assert X_train.shape[0] == y_train.shape[0], "Number of samples in X and y do not match"
        pipeline = create_pipeline()
        pipeline.set_params(**best_params)
        pipeline.fit(X_train, y_train)
        return pipeline
    except AssertionError as e:
        print(e)
    except Exception as e:
        print("Error occured while training model: ", e)


def evaluate_model(pipeline, X, y):
    """
    This function evaluates the performance of the given pipeline on the given data
    :param pipeline: pipeline object to evaluate
    :type pipeline: sklearn.pipeline.Pipeline
    :param X: input feature variables
    :type X: array-like
    :param y: target variable
    :type y: array-like
    :return: mean squared error, mean absolute error
    :rtype: tuple
    """
    try:
        assert pipeline is not None, "Pipeline object should be initialized before evaluation"
        y_pred = pipeline.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return mse, mae
    except AssertionError as ae:
        print(ae)
    except Exception as e:
        print(f"An error occurred while evaluating the model: {e}")


def save_metrics(mse, mae, filepath):
    """
    Save the accuracy and confusion matrix to a local file
    """
    try:
        assert filepath is not None, "filepath should be provided"
        with open(filepath, "w") as f:
            f.write("Mean Squared Error: {:.4f}\n".format(mse))
            f.write("Mean Absolute Error: {:.4f}\n".format(mae))
    except AssertionError as ae:
        print(ae)
    except Exception as e:
        print(f"An error occurred while saving the metrics to a file: {e}")


def challenge_3():
    """
    This function performs the following tasks:
    1. Loads the wine quality dataset from sklearn
    2. Splits the data into training and testing sets
    3. Creates a pipeline with a RandomForestRegressor and StandardScaler
    4. Hyperparameter tunes the pipeline using GridSearchCV
    5. Trains the pipeline on the training data
    6. Evaluates the pipeline on the testing data
    7. Saves the evaluation metrics to a local file
    """
    parser = argparse.ArgumentParser(description='Challenge 2')
    parser.add_argument('filepath', type=str, help='filepath to save metrics to')
    args = parser.parse_args()
    filepath = args.filepath
    try:
        # Load the wine quality dataset
        print("Loading Wine Quality dataset...")
        X, y = load_wine(return_X_y=True)
        print("Wine Quality dataset loaded.")

        # Split the data into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print("Data split.")

        # Create the pipeline
        print("Creating pipeline...")
        pipeline = create_pipeline()
        print("Pipeline created.")

        # Tune the hyperparameters
        print("Starting hyperparameter tuning...")
        best_params = tune_hyperparameters(pipeline, X_train, y_train)
        print("Hyperparameter tuning complete.")

        # Train the model
        print("Training model...")
        model = train_model(X_train, y_train, best_params)
        print("Model trained.")
        # Evaluate the model
        print("Evaluating model...")
        mse, mae = evaluate_model(model, X_test, y_test)
        print("Model evaluated.")

        # Save the evaluation metrics to a local file
        print("Saving Metrics")
        save_metrics(mse, mae, filepath)
        print("Metrics Saved")
    except Exception as e:
        print("An error occurred: ", e)
        raise e


if __name__ == '__main__':
    challenge_3()
 