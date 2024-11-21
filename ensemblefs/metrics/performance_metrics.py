from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


class BaseMetric:
    def __init__(self, name, task="classification"):
        """
        Initialize BaseMetric with a task.
        Args:
            name (str): Name of the metric.
            task (str): Task type, either 'classification' or 'regression'.
        """
        self.name = name
        self.task = task
        self.classifiers = self._initialize_models()

    def _initialize_models(self):
        """
        Initialize models based on the task.
        Returns:
            dict: A dictionary of models appropriate for the task.
        """
        if self.task == "classification":
            return {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, solver="lbfgs", penalty="l2"
                ),
                "Gradient Boosting": GradientBoostingClassifier(),
            }
        elif self.task == "regression":
            return {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
            }
        else:
            raise ValueError("Invalid task. Choose 'classification' or 'regression'.")

    def train_and_predict(self, X_train, y_train, X_test, y_test):
        """
        Train all models on X_train, y_train and generate predictions.
        Returns a dictionary with predictions for regression and
        predictions + probabilities for classification.
        """
        results = {}

        for model_name, model in self.classifiers.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = None
            if self.task == "classification":
                probabilities = model.predict_proba(X_test)
            results[model_name] = {
                "predictions": predictions,
                "probabilities": probabilities,
            }

        return results

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Abstract method to compute the metric.
        Must be implemented in child classes.
        """
        raise NotImplementedError("This method should be overridden in child classes.")


class R2Score(BaseMetric):
    def __init__(self):
        super().__init__(name="R2 Score", task="regression")

    def compute(self, X_train, y_train, X_test, y_test):
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for model_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(r2_score(y_test, predictions))

        return sum(metric_values) / len(metric_values)


class MeanAbsoluteError(BaseMetric):
    def __init__(self):
        super().__init__(name="Mean Absolute Error", task="regression")

    def compute(self, X_train, y_train, X_test, y_test):
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for model_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(mean_absolute_error(y_test, predictions))

        return -sum(metric_values) / len(metric_values)  # Return negative MAE


class MeanSquaredError(BaseMetric):
    def __init__(self):
        super().__init__(name="Mean Squared Error", task="regression")

    def compute(self, X_train, y_train, X_test, y_test):
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for model_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(mean_squared_error(y_test, predictions))

        return -sum(metric_values) / len(metric_values)  # Return negative MSE


# Child Classes
class LogLoss(BaseMetric):
    def __init__(self):
        super().__init__(name="Log Loss")

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Compute the average Log Loss across all classifiers.
        """
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for clf_name, result in results.items():
            probabilities = result["probabilities"]
            metric_values.append(log_loss(y_test, probabilities))

        return sum(metric_values) / len(metric_values)


class F1Score(BaseMetric):
    def __init__(self):
        super().__init__(name="F1Score")

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Compute the average Macro F1 Score across all classifiers.
        """
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for clf_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(f1_score(y_test, predictions, average="macro"))

        return sum(metric_values) / len(metric_values)


class Accuracy(BaseMetric):
    def __init__(self):
        super().__init__(name="Accuracy")

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Compute the average Accuracy across all classifiers.
        """
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for clf_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(accuracy_score(y_test, predictions))

        return sum(metric_values) / len(metric_values)


class PrecisionScore(BaseMetric):
    def __init__(self):
        super().__init__(name="Precision Score")

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Compute the average Precision Score across all classifiers.
        """
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for clf_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(
                precision_score(y_test, predictions, average="macro", zero_division=0)
            )

        return sum(metric_values) / len(metric_values)


class RecallScore(BaseMetric):
    def __init__(self):
        super().__init__(name="Recall Score")

    def compute(self, X_train, y_train, X_test, y_test):
        """
        Compute the average Recall Score across all classifiers.
        """
        results = self.train_and_predict(X_train, y_train, X_test, y_test)
        metric_values = []

        for clf_name, result in results.items():
            predictions = result["predictions"]
            metric_values.append(recall_score(y_test, predictions, average="macro"))

        return sum(metric_values) / len(metric_values)
