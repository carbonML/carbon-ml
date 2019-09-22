from unittest import TestCase, main
from mock import patch, MagicMock
from carbon_ml.atom.inputs.input_data import InputDataFrame
import pandas as pd


class TestInputDataFrame(TestCase):

    @patch("carbon_ml.atom.inputs.input_data.isinstance")
    def test___init__(self, mock_isinstance):
        mock_isinstance.return_value = True
        InputDataFrame(data_frame="test")
        mock_isinstance.assert_called_once_with("test", pd.DataFrame)

        mock_isinstance.return_value = False

        with self.assertRaises(Exception):
            InputDataFrame(data_frame="test")

    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.__init__")
    def test_check_columns(self, mock_init):
        mock_init.return_value = None
        data_frame = pd.DataFrame([{"one": 1, "two": 2, "three": 3}])
        test = InputDataFrame(data_frame="test")
        test.data = data_frame

        test.check_columns(check="one")
        test.check_columns(check=["one"])
        test.check_columns(check=["one", "two"])

        with self.assertRaises(Exception):
            test.check_columns(check="test")
        with self.assertRaises(Exception):
            test.check_columns(check=["test"])
        with self.assertRaises(Exception):
            test.check_columns(check=["one", "test"])

    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.check_columns")
    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.scale_data")
    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.__init__")
    def test_prep_for_training(self, mock_init, mock_scale_data, mock_check_columns):
        mock_init.return_value = None
        data_frame = pd.DataFrame(
            [
                {"one": 1, "two": 2, "three": 3},
                {"one": 1, "two": 2, "three": 3},
                {"one": 1, "two": 2, "three": 3},
                {"one": 1, "two": 2, "three": 3},
                {"one": 1, "two": 2, "three": 3}
            ]
        )

        test = InputDataFrame(data_frame="test")
        test.data = data_frame
        test.prep_for_training(outcome_pointer="one", params_dict={})
        self.assertEqual(["two", "three"], list(test.x_train))
        self.assertEqual(["two", "three"], list(test.x_test))
        self.assertEqual([1, 1, 1], list(test.y_train))
        self.assertEqual([1, 1], list(test.y_test))
        mock_check_columns.assert_called_once_with(check="one")

        test.prep_for_training(outcome_pointer=["one", "two"], params_dict={})
        self.assertEqual(["one", "two"], list(test.y_train))
        self.assertEqual(["one", "two"], list(test.y_test))
        self.assertEqual(["three"], list(test.x_train))
        self.assertEqual(["three"], list(test.x_test))

        test.prep_for_training(outcome_pointer="one", params_dict={"scaling_tool": "test scaler"})
        mock_scale_data.assert_called_once_with(scaling_tool="test scaler")

    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.SCALING_TOOLS")
    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.__init__")
    def test_scale_data(self, mock_init, mock_scaling_dict):
        mock_init.return_value = None
        test = InputDataFrame(data_frame="test")
        test.x_train = MagicMock()
        x_train_placeholder = test.x_train
        test.x_test = MagicMock()
        mock_scaling_dict.get.return_value = MagicMock()
        test.scale_data(scaling_tool="standard")

        mock_scaling_dict.get.assert_called_once_with("standard")
        test.scaling_tool.fit.assert_called_once_with(x_train_placeholder)

        mock_scaling_dict.get.return_value = None
        with self.assertRaises(Exception):
            test.scale_data(scaling_tool="test")

    @patch("carbon_ml.atom.inputs.input_data.InputDataFrame.__init__")
    def test_prep_inputs(self, mock_init):
        mock_init.return_value = None
        test = InputDataFrame(data_frame="test")
        test.input_order = None

        with self.assertRaises(Exception):
            test.prep_inputs(input_dict={"one": 1, "two": 2, "three": 3})

        test.input_order = ["one", "two", "three"]
        with self.assertRaises(Exception):
            test.prep_inputs(input_dict={"one": 1, "two": 2, "four": 3})

        out_come = test.prep_inputs(input_dict={"one": 1, "two": 2, "three": 3})
        self.assertEqual([1, 2, 3], out_come)


if __name__ == "__main__":
    main()
