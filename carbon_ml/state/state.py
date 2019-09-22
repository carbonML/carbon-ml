from carbon_ml.caching import CacheManager


class StateOfMatter:

    def __init__(self):
        self.model = None
        self.data = None
        self.cache = CacheManager()

    def load(self):
        pass

    def save(self):
        pass

    def new(self):
        pass

    def calculate(self):
        pass
