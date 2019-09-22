from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from carbon_ml.atom.inputs.input_data import InputDataFrame
from carbon_ml.atom.errors import AtomError
from carbon_ml.atom.plotting.basic import BasicPlotter


class Atom:
    """
    This class manages the data and machine learning algorithm training.

    Attributes:
        data (carbon_ml.atom.inputs.input_data.InputDataFrame): data to be trained on
        model (sk-learn model): model to be trained
        train_errors (list): training errors
        test_errors (list): testing errors
    """
    def __init__(self, data, model, name):
        """
        The constructor for the Atom class.

        :param data: (pandas data frame) data to be trained on
        :param model: (sklearn model) model to be trained
        :param name: (str) name of Atom object
        """
        self.data = InputDataFrame(data_frame=data)
        self.model = model
        self.name = name
        self.train_errors = []
        self.test_errors = []
        self.classification_report = None

    def train(self, params_dict):
        """
        Trains the self.model.

        :param params_dict: (dict) params for training
        :return: None
        """
        self.train_errors = []
        self.test_errors = []
        self.data.prep_for_training(outcome_pointer=params_dict.get("outcome_pointer"), params_dict=params_dict)
        cut_off = params_dict.get("cut_off", False)

        for i in range(params_dict.get("starting_point", 100),
                       len(self.data.x_train), params_dict.get("batch_size", 100)):
            self.model.fit(self.data.x_train[:i], self.data.y_train[:i])

            y_train_predict = self.model.predict(self.data.x_train[:i])
            y_test_predict = self.model.predict(self.data.x_test)

            self.train_errors.append(mean_squared_error(y_train_predict, self.data.y_train[:i]))
            self.test_errors.append(mean_squared_error(y_test_predict, self.data.y_test))

            if cut_off is not False:
                if len(self.train_errors) == cut_off:
                    break
        if params_dict.get("evaluate") == "classification":
            self.evaluate_classification_performance()
            print(self.classification_report)

        BasicPlotter.show_learning_curve(train_errors=self.train_errors, test_errors=self.test_errors)

    def predict_probability(self, input_dict):
        """
        Gives probability of outcomes.

        :param input_dict: (list) inputs for model
        :return: array of probabilities
        """
        input_array = self.data.prep_inputs(input_dict=input_dict)
        return self.model.predict_proba([input_array])

    def evaluate_classification_performance(self):
        """
        Produces classification report of finished model
        :return: list of predictions from the X_test data subset
        """
        predictions = self.model.predict(self.data.x_test)
        self.classification_report = classification_report(self.data.y_test, predictions)

    def deploy(self):
        # overwrite the picking functions
        pass
