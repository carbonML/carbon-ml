from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from carbon_ml.atom.errors import InputDataFrameError
import pandas as pd


class InputDataFrame:
    """
    This is the class managing the input data for training.

    Attributes:
        data (pandas data frame): data to be processed and fed into algorithm
        input_order (list): list of inputs for the algorithm
        x_train (pandas data frame) input training data
        x_test (pandas data frame): input testing data
        y_train (pandas data frame): output training data
        y_test (pandas data frame): output testing data
        scaling_tool (obj): tool used to scale data
    """
    SCALING_TOOLS = {
        "standard": StandardScaler(),
        "min max": MinMaxScaler()
    }

    def __init__(self, data_frame):
        """
        The constructor for the InputDataFrame class.

        :param data_frame: (pandas data frame) data to be processed
        """
        if isinstance(data_frame, pd.DataFrame) is False:
            raise InputDataFrameError("{} passed instead of pandas data frame".format(type(data_frame)))
        self.data = data_frame
        self.input_order = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.scaling_tool = None

    @property
    def columns(self):
        return list(self.data)

    def check_columns(self, check):
        """
        Checks columns are in self.data.

        :param check: (str/list) column to be checked
        :return: None
        """
        if isinstance(check, str):
            if check not in self.columns:
                raise InputDataFrameError(message="{} not in data frame".format(check))
        if isinstance(check, list):
            for column in check:
                if column not in self.columns:
                    raise InputDataFrameError(message="{} not in data frame".format(check))
        
    def prep_for_training(self, outcome_pointer, params_dict):
        """
        Preps self.data for machine learning.

        :param outcome_pointer: (list/str) columns
        :param params_dict: (dict) dict to define if scaling is needed
        :return: None
        """
        self.check_columns(check=outcome_pointer)
        x = self.data.drop(outcome_pointer, axis=1)
        y = self.data[outcome_pointer]
        self.input_order = list(x.columns.values)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y,
            test_size=params_dict.get("test_size", 0.33),
            random_state=params_dict.get("random_state", 101))

        if params_dict.get("scaling_tool"):
            self.scale_data(scaling_tool=params_dict.get("scaling_tool"))

    def scale_data(self, scaling_tool):
        """
        Scales the data after it's been split.

        :param scaling_tool: (str) name of scaler selected
        :return: None
        """
        self.scaling_tool = self.SCALING_TOOLS.get(scaling_tool)
        if scaling_tool is None:
            raise InputDataFrameError("{} is not a scaling tool, please pick between: {}".format(
                scaling_tool, self.SCALING_TOOLS.keys()
            ))

        self.scaling_tool.fit(self.x_train)
        self.x_train = self.scaling_tool.transform(self.x_train)
        self.x_test = self.scaling_tool.transform(self.x_test)
