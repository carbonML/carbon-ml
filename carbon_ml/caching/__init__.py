from carbon_ml.caching.errors import CacheManagerError
from carbon_ml.caching.worker import Worker
from carbon_ml.caching.singleton import Singleton
from carbon_ml.caching.mapping.soft_map import SoftMap


class CacheManager(metaclass=Singleton):

    def __init__(self):
        """
        The constructor for the CacheManager class.
        """
        self.soft_cache = SoftMap()

    def add_cache(self, name):
        """
        Adds Worker object to self.soft_cache

        :param name: (str) name of the cache
        :return: None
        """
        self.soft_cache[name] = Worker()

    def get_cache_path(self, name):
        """
        Gets the path to the cache.

        :param name: (str) name of the cache
        :return: (str) path to the cache
        """
        if name not in self.soft_cache.keys():
            raise CacheManagerError(message="{} not in cache".format(name))
        return self.soft_cache[name].base_dir

    def wipe_cache(self):
        """
        Wipes the cache.

        :return: None
        """
        self.soft_cache.wipe_map()

    def __enter__(self):
        self.add_cache(name="carbon's temp cache")

    def __exit__(self, type, value, traceback):
        del self.soft_cache["carbon's temp cache"]
