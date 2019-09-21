from unittest import TestCase, main
from carbon_ml.caching.mapping.soft_map import SoftMap
from carbon_ml.caching.worker import Worker


class TestSoftMap(TestCase):

    def test___init__(self):
        test = SoftMap()
        test_two = SoftMap()

        self.assertEqual({}, test)
        self.assertEqual(id(test), id(test_two))

    def test_add_worker(self):
        test = SoftMap()

        with self.assertRaises(Exception):
            test["test"] = "fake"

        first_worker = Worker()
        second_worker = Worker()

        test["test"] = first_worker

        self.assertEqual({"test": first_worker}, test)

        with self.assertRaises(Exception):
            test["test"] = first_worker
        self.assertEqual({"test": first_worker}, test)

        with self.assertRaises(Exception):
            test["test two"] = first_worker
        self.assertEqual({"test": first_worker}, test)

        test["test two"] = second_worker
        self.assertEqual({"test": first_worker, "test two": second_worker}, test)

    def test_wipe_map(self):
        test = SoftMap()
        self.assertTrue("test" in test.keys())
        self.assertTrue("test two" in test.keys())
        test.wipe_map()
        self.assertEqual({}, test)


if __name__ == "__main__":
    main()
