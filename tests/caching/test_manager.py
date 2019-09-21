from unittest import TestCase, main
from mock import patch, MagicMock
from carbon_ml.caching import CacheManager


class TestCacheManager(TestCase):

    def test___init__(self):
        # initiate the manager object
        test = CacheManager()

        # create a new instance of the object
        test_two = CacheManager()

        # check to see if the memory location is the same, if so they're the same object
        self.assertEqual(id(test), id(test_two))

    @patch("carbon_ml.caching.Worker")
    def test_add_cache(self, mock_worker):
        test = CacheManager()
        test.soft_cache = MagicMock()

        test.add_cache(name="test name")
        test.soft_cache.__setitem__.assert_called_once_with("test name", mock_worker.return_value)

    def test_get_cache_path(self):
        test = CacheManager()
        test.soft_cache = {"one": MagicMock(), "two": MagicMock()}

        with self.assertRaises(Exception):
            test.get_cache_path(name="three")

        out_come = test.get_cache_path(name="one")
        self.assertEqual(test.soft_cache["one"].base_dir, out_come)

    def test_wipe_cache(self):
        # initiate the manager object
        test = CacheManager()

        test.soft_cache = MagicMock()
        test.wipe_cache()

        test.soft_cache.wipe_map.assert_called_once_with()

    @patch("carbon_ml.caching.CacheManager.add_cache")
    def test___enter__(self, mock_add_cache):
        # initiate the manager object
        test = CacheManager()
        test.soft_cache = {"carbon's temp cache": "test"}

        # run a session block
        with test:
            self.assertEqual({"carbon's temp cache": "test"}, test.soft_cache)

        # ensure that a cache was created
        mock_add_cache.assert_called_once_with(name="carbon's temp cache")

        # ensure that a cache was deleted
        self.assertEqual({}, test.soft_cache)


if __name__ == "__main__":
    main()
