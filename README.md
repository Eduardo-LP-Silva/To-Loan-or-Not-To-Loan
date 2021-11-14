# ECAC-Competition
Scripts developed for the "Knowledge Extraction and Machine Learning" (ECAC) class "To Loan or Not To Loan" data mining case study / Kaggle competition:

* data_understanding.py - Contains methods to analyse the multiple data sources and provide visual representations of relevant patterns, as well as to calculate various statistics regarding certain attributes.
* data_preparation.py - Contains methods to pre-process the data, including filling missing values with the previously calculated statistics, removing correlated attributes and outliers, one-hot encoding categorical features and normalizing the data.
* k_nn.py / rf.py / svm.py - Contains the methods associated with each algorithm (K-NN, random forest, SVM) to split and balance the data, perform hyper-parameter optimizations and perform the final class predictions.
