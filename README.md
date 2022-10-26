


A module for preparing and processing data from the development of a mineral deposit. Based on 
machine learning methods, it allows data preprocessing (data sampling over space and time, 
clustering, etc.), as well as calculating deformation fields (in progress).

Collected materials (datasets in csv files) and source code (Python programming language).

## Requirements
    'python>=3.7'
    'scikit-learn==0.23.2'
    'numpy'
    'pandas'
    'seaborn'
    'scipy'
    'pickle'
    'datetime'
    'matplotlib'
    'hdbscan'
    'glob'
    'tensorly'
    
## Documentation
The module is designed for processing data in tabular form (e.g. pandas dataframes). At the
 moment, there are several main blocks within which you can process data:
- [Preprocessing. Clusterization](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/docs/cluster.md);
- [Preprocessing. Grid](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/docs/grid.md);
- [Launch](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/docs/launcher.md).

## Examples
- [Low-level functions example](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/examples/simple_example.py);
- [High-level launcher example](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/examples/launcher_example.py).

## Tutorials
- [Launcher tutorial (in russian)](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/examples/launcher_tutorial.ipynb)