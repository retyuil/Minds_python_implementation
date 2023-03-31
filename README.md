# Minds_python_implementation
 Minds intrusion dection algorithme made in python

Python code based on this article : https://www.sciencedirect.com/science/article/pii/S0167404814000923

Dataset used for the algorithm : https://www.stratosphereips.org/datasets-ctu13

Two version of the algorithm : 
-One with the number of adjacent netflows taken into account (Minds_with_features.py)
-Seconde one without feature it take into account all netflow into the dataset for each netflow studied

# Usage : 

Install the lib needed in requirement.txt : 

```sh
pip install requirement.txt

```

Modify the name of you dataset file in the source code.

```sh
python Minds_with_features

```

```sh
python Minds_without_features

```

