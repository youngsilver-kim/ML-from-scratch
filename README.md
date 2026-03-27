# ML-from-scratchml-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_linear_regression_demo.ipynb
│   ├── 02_logistic_regression_demo.ipynb
│   ├── 03_knn_demo.ipynb
│   ├── 04_decision_tree_demo.ipynb
│   ├── 05_kmeans_demo.ipynb
│   └── 06_pca_demo.ipynb
├── ml_scratch/
│   ├── __init__.py
│   ├── base.py
│   ├── metrics.py
│   ├── model_selection.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── linear_model/
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   └── logistic_regression.py
│   ├── neighbors/
│   │   ├── __init__.py
│   │   └── knn.py
│   ├── tree/
│   │   ├── __init__.py
│   │   └── decision_tree.py
│   ├── cluster/
│   │   ├── __init__.py
│   │   └── kmeans.py
│   └── decomposition/
│       ├── __init__.py
│       └── pca.py
├── tests/
│   ├── test_linear_regression.py
│   ├── test_logistic_regression.py
│   ├── test_knn.py
│   ├── test_decision_tree.py
│   ├── test_kmeans.py
│   └── test_pca.py
├── examples/
│   ├── linear_regression_example.py
│   ├── logistic_regression_example.py
│   ├── knn_example.py
│   ├── decision_tree_example.py
│   ├── kmeans_example.py
│   └── pca_example.py
└── assets/
    ├── linear_regression_result.png
    ├── logistic_boundary.png
    ├── knn_visualization.png
    ├── decision_tree_result.png
    ├── kmeans_clusters.png
    └── pca_projection.png
