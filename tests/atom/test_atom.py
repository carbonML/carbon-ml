from unittest import TestCase, main
from mock import patch, MagicMock
from carbon_ml.atom import Atom


class TestAtom(TestCase):

    @patch("carbon_ml.atom.InputDataFrame")
    def test___init__(self, mock_input_data):
        test = Atom(data="test data", model="test model", name="test name")
        mock_input_data.assert_called_once_with(data_frame="test data")

        self.assertEqual(mock_input_data.return_value, test.data)
        self.assertEqual("test model", test.model)
        self.assertEqual("test name", test.name)
        self.assertEqual([], test.train_errors)
        self.assertEqual([], test.test_errors)

    @patch("carbon_ml.atom.mean_squared_error")
    @patch("carbon_ml.atom.Atom.__init__")
    def test_train(self, mock_init, mock_mean_squared_error):
        mock_init.return_value = None
        test = Atom(data="test data", model="test model", name="test name")
        test.data = MagicMock()
        test.model = MagicMock()

        test.data.x_train = [1, 2, 3, 4, 5]
        test.data.y_train = [1, 2, 3, 4, 5]
        test.data.x_test = [1, 2, 3, 4, 5]
        test.data.y_test = [1, 2, 3, 4, 5]

        test.train(params_dict={"starting_point": 1, "batch_size": 1})

        self.assertEqual(([1], [1]), test.model.fit.call_args_list[0][0])
        self.assertEqual(([1, 2], [1, 2]), test.model.fit.call_args_list[1][0])
        self.assertEqual(([1, 2, 3], [1, 2, 3]), test.model.fit.call_args_list[2][0])
        self.assertEqual(([1, 2, 3, 4], [1, 2, 3, 4]), test.model.fit.call_args_list[3][0])

        self.assertEqual(([1],), test.model.predict.call_args_list[0][0])
        self.assertEqual(([1, 2, 3, 4, 5],), test.model.predict.call_args_list[1][0])
        self.assertEqual(([1, 2],), test.model.predict.call_args_list[2][0])
        self.assertEqual(([1, 2, 3, 4, 5],), test.model.predict.call_args_list[3][0])

        self.assertEqual(8, len(mock_mean_squared_error.call_args_list))

        test.train(params_dict={"starting_point": 1, "batch_size": 2})

        self.assertEqual(12, len(mock_mean_squared_error.call_args_list))


if __name__ == "__main__":
    main()
