# Caching
This where the caching logic is stored. Carbon-ml is based on this caching 
mechanism. When a training session is done, the dataset, trained model and 
meta data is stored in the cache. This can free up the memory for the computer.
```__del__``` functions are overridden so if the computer crashes or the program finishes the cache will be cleared. Exporting the project will save
it in a hard cache so it can be accessed whenever. 

## Structure 
The caching mechanism takes a flat structure at the moment but can change as the project evolves:

```
├── README.md
├── __init__.py
├── cache
│   └── __init__.py
├── errors.py
├── hard_cache
│   └── __init__.py
├── mapping
│   └── soft_map.py
├── singleton.py
└── worker.py
```
The ```singleton.py``` file houses the singleton class which ensures that 
any class using it as a metaclass cannot be redefined or ducplicated. Temp 
projects get stored in the cache directory. Hard cache does not get wiped 
at every runtime. The cache manager is stored in the ```__init__.py```

## Using 
Importing and using the cache manager only takes a few lines with no external 
dependancies:

```python
from carbon_ml.caching import CacheManager

test = CacheManager()
test.add_cache("testing")
test.get_cache_path("testing")
```
