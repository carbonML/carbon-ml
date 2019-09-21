import uuid
import os
import shutil
from carbon_ml.caching.errors import WorkerCacheError


class Worker:
    """
    This is a class for managing a directory for temp files

    Attributes:
        id (str): unique id for the worker
        base_dir (str): directory path for the cache 
    """

    CLASS_BASE_DIR = os.path.realpath(__file__)
    CLASS_BASE_DIR = CLASS_BASE_DIR.replace("worker.py", "")

    def __init__(self):
        """
        The constructor for the Worker class.

            Parameters:
                None
        """
        self.id = uuid.uuid1()
        self.base_dir = str(self.CLASS_BASE_DIR) + "cache/{}/".format(self.id)
        self._generate_directory()

    def _generate_directory(self):
        """
        Generates cache directory with self.id (private).

        :return: None
        """
        if os.path.isdir(self.base_dir):
            raise WorkerCacheError(message="directory {} already exists. Check __del__ and self.id methods".format(
                self.base_dir
            ))
        os.mkdir(self.base_dir)

    def _delete_directory(self):
        """
        Deletes cache directory (private).

        :return: None
        """
        if not os.path.isdir(self.base_dir):
            raise WorkerCacheError(
                "directory {} does not exist. please check __del__ and self._delete_directory methods".format(
                    self.base_dir))
        shutil.rmtree(self.base_dir)

    def __del__(self):
        """
        Fires when self is deleted, deletes the directory.

        :return: None
        """
        self._delete_directory()
