from carbon_ml.caching.worker import Worker
from carbon_ml.caching.errors import SoftMapError
from carbon_ml.caching.singleton import Singleton


class SoftMap(dict, metaclass=Singleton):

    def __init__(self):
        super().__init__({})

    def __setitem__(self, key, value):
        """
        Overwrites the set item function to ensure that only Worker objects can be added

        :param key: key to map to object
        :param value: object to be stored
        :return: None
        """
        type_check = Worker()
        if type(type_check) != type(value):
            raise SoftMapError(message="{} tried to be added to SoftMap".format(type(value)))
        del type_check

        if key in self.keys():
            raise SoftMapError(message="{} namespace already exists in soft cache map".format(key))

        for name in self.keys():
            if str(self[name].id) == str(value.id):
                raise SoftMapError(message="{} for {} already exists in SoftMap".format(value.id,
                                                                                        name))

        super().__setitem__(key, value)

    def wipe_map(self):
        """
        Wipes the map clean.

        :return: None
        """
        keys = list(self.keys())
        for key in keys:
            del self[key]

    def display(self):
        pass
