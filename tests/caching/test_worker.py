from unittest import TestCase, main
from mock import patch
from carbon_ml.caching.worker import Worker


class TestWorker(TestCase):

    @patch("carbon_ml.caching.worker.Worker._delete_directory")
    @patch("carbon_ml.caching.worker.Worker._generate_directory")
    @patch("carbon_ml.caching.worker.uuid.uuid1")
    def test___init__(self, mock_uuid, mock_generate_directory, mock_delete_directory):
        # define the return value for the mocked uuid
        mock_uuid.return_value = "test"

        # run the test by initialising the worker
        test = Worker()

        # check that the test.id is the same as the return value from the uuid module
        self.assertEqual(mock_uuid.return_value, test.id)

        # delete the test object now instead of waiting for the end of the program to do it
        del test

        # check to see if the generate directory function is called once with nothing
        mock_generate_directory.assert_called_once_with()

        # check to see if the delete directory function is called once with nothing
        mock_delete_directory.assert_called_once_with()

    @patch("carbon_ml.caching.worker.os")
    @patch("carbon_ml.caching.worker.Worker._delete_directory")
    @patch("carbon_ml.caching.worker.Worker.__init__")
    def test__generate_directory(self, mock_init, mock_delete_directory, mock_os):
        # setup return values for function to pass
        mock_init.return_value = None
        mock_os.path.isdir.return_value = False

        # initiate the worker and run the function
        test = Worker()
        test.base_dir = "test dir"
        test._generate_directory()

        # check that the checking of path has been done once with the base dir
        mock_os.path.isdir.assert_called_once_with(test.base_dir)

        # check that the mkdir function is called once with the base dir
        mock_os.mkdir.assert_called_once_with(test.base_dir)

        # setup return values for the directory to already exist
        mock_os.path.isdir.return_value = True

        # check that it raises an exception because of this
        with self.assertRaises(Exception):
            test._generate_directory()

        # check that the mkdir has still only been called once
        mock_os.mkdir.assert_called_once_with(test.base_dir)

        # check that the isdir function has been called twice
        self.assertEqual(2, len(mock_os.path.isdir.call_args_list))

        del test

        mock_delete_directory.assert_called_once_with()

    @patch("carbon_ml.caching.worker.shutil")
    @patch("carbon_ml.caching.worker.os")
    @patch("carbon_ml.caching.worker.Worker.__init__")
    def test__delete_directory(self, mock_init, mock_os, mock_shutil):
        # setup return values for function to pass
        mock_init.return_value = None
        mock_os.path.isdir.return_value = True

        # initiate the worker and run the function
        test = Worker()
        test.base_dir = "test dir"
        test._delete_directory()

        # check that the checking of path has been done once with the base dir
        mock_os.path.isdir.assert_called_once_with(test.base_dir)

        # check that the rmtree function is called once with the base dir
        mock_shutil.rmtree.assert_called_once_with(test.base_dir)

        # setup return values for the directory to not exist
        mock_os.path.isdir.return_value = False

        # check that it raises an exception because of this
        with self.assertRaises(Exception):
            test._delete_directory()

        # check that the rmtree has still only been called once
        mock_shutil.rmtree.assert_called_once_with(test.base_dir)

        # check that the isdir function has been called twice
        self.assertEqual(2, len(mock_os.path.isdir.call_args_list))

        # set to true before exiting to avoid error raising when exiting program
        mock_os.path.isdir.return_value = True


if __name__ == "__main__":
    main()
