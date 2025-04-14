# A general framework to govern machine learning oriented materials data quality (MAT-DQG)

**Traceability:** Meta-data records the information of  data itself as well as the process of data acquisition and processing.

| Metadata            | Types     | Field Name       | Metadata              | Field Name                     |
| ------------------- | --------- | ---------------- | --------------------- | ------------------------------ |
|                     | Dimension | Descriptor  Name |                       |                                |
|                     | Dimension | Property         |                       | Tasks of Machine learning      |
|                     | Sample    | Sample Name      |                       |                                |
|                     | Sample    | Sample  source   |                       | Method of Data-preprocessing   |
|                     | Sample    | Acquisition      |                       |                                |
| **Basic meta-data** | Dataset   | ID               | **Derived meta-data** | Method of feature engineering  |
|                     | Dataset   | Scale            |                       |                                |
|                     | Dataset   | Material  types  |                       | Method of dataset partitioning |
|                     | Dataset   | Dataset source   |                       |                                |
|                     | Dataset   | Summary          |                       | Algorithms and performances    |
|                     | Dataset   | DOI              |                       |                                |
|                     | Dataset   | Contributor      |                       | Hyperparameter                 |
|                     | Dataset   | AddTime          |                       |                                |

**Time-sensitivity**\
​*​Permutation entropy.py:​*\
​To determine the minimum embedding dimension for time series data based on permutation entropy. According to the computed results, each temporal feature of the dataset is partitioned, and the classification outcomes of the features are automatically stored.

**Balance**\
​*​Clustering for Balance.py:​*\
​Clustering-based algorithm (e.g., K-means) for clustering the samples with all features (descriptors) and target property, only all features and only target property, respectively.

**Consistency**\
​*Consistency detection:​*\
​		(1) ConsistencyDetection.py: Evaluate whether all samples in the material data were uniformly captured or characterized under the same conditions. First, determine whether the units of measurement for each feature in every column are consistent, and standardize them if inconsistencies are found. Then, assess whether the representation formats of features within each column are uniform, applying standardization if discrepancies are detected.\
​		(2) Processing.py: Processing the tabular data.

**Completeness**\
*​CompletenessDetection.py:*\
​Evaluate the completeness of characteristic values pertaining to the material data itself, determining whether the dataset contains missing values. If missing values are present, the department first calculates the missing data rate for the affected data. Subsequently, multiple imputation methods are employed to fill in the missing data. Upon obtaining a complete dataset, it is automatically saved.

**Normalization**\
​*NormalizationDetection.py:*\
​		(1) Numerical normalization offers a variety of data standardization methods to mitigate the adverse effects of different dimensions and numerical ranges.\
​		(2) Partitioning normalization recommends appropriate dataset partitioning methods based on the scale of the dataset, enabling the model to learn the characteristics of the data during the training process and thereby enhancing the model's generalization capability.

**Accuracy**\
*​Outlier detection:*\
​    	(1) SingleDimensionalOutlierDetection.py: Single-Dimensional Outlier Detection (Quartile Method);\
   	(2) BoxPlotting.py: Plotting Box Plots (Schematic Diagram of Quartile Results);\
   	(3) MultiDimensionalOutlierDetection.py: Multidimensional outlier sample detection;\
 	   (4) AllDimensionalOutlierDetection.py: Scatter plots are plotted based on clustering of  the samples with all features (descriptors) and target property and target property.

**Redundancy**\
​*Feature selection and machine learning model prediction:*\
​		(1) FeatureSelection/MBPSO.py: The main algorithm of the NCOR-FS method (including population initialization and evolution);\
​		(2) FeatureSelection/NondominatedSolutionCal.py: Non-dominant solution recognition;\
​		(3) FeatureSelection/ViolationDegreeCal.py: Calculation of NCOR Violation;\
​		(4) FeatureSelection/DncorCal: Highly correlated feature recognition.

**Insight**\
​​*General program:​*\
​		(1) BayesOptModels.py: A variety of machine-learned regression models for modeling and prediction;\
​		(2) Prediction.py: Machine learning modeling based on the original dataset and the corrected dataset;\
​		(3) utiles.py: Encapsulates functions such as creating directories and saving files.
