# Word-Sense-Disambiguation
## Introduction
There are many words in our day-to-day life which have similar spelling with different meaning. For example, word ‘cold’ represents two meaning, one is ‘absent of heat’ while second is mostly used during winter for nose blocking. The problem of determining the context for given target word from the sentence is called word sense disambiguation in Natural language processing (NLP).In this repo, we are using support vector machine (SVM) technique to classify various context. We used ‘SENSEVAL’ dataset to train the SVM classifier. This dataset offer corpus for four different words: line, interest, hard and serve. We trained SVM for ‘line’ word for different parameters and come up with best parameter that can classify the context with highest accuracy.
## Setup for training
* Open the [Training_main.py](Training_main.py) file and change arguments that are listed below.
  * Target_word: Give the target word for that you want to train model.
  * Text_path: Give path for your training text file
  * Result_path: Give path for directory where you want to store results.
  * Label: Give the list of senses for your target word that are defined in dataset. Give the integer number to each sense as shown in code.
  * Kernel_type: Give your kernel name (‘poly’ or ‘linear’ or ‘rbf’).
  * C_value: Give the list of C values. The given values in code is already covering all values. But you can add or remove.
  * Cross_validation: Give the number of cross validation you want during training.
* If you are using SENSEVAL dataset, then you do not need to change except above arguments. But if you are using any other dataset, than you have to do little bit modification in [Data_collect.py](Data_collect.py) according to dataset format.
* Run [Training_main.py](Training_main.py) file.
* At the end of training, you will see four files in Result directory.
  * <target_word>_analysis.txt: It has all training result and evaluation result.
  * <target_word>_vocabulary.csv: It has list of all context words those are used as feature during training.
  * <target_word>_weight.pkl: It is model which is used during testing.
  * <target_word>_vectorizer.pkl: It is used to create feature for testing data during testing.
## Setup for testing
* Open the [Testing.py](Testing.py) file and change arguments that are listed below.
  * Model_path: Give the path for “<target_word>_weight.pkl” file which was generated after training.
  * Vectorizer_path: Give the path for “<target_word>_vectorizer.pkl” file which was generated after training.
  * Test_path: Give the path for testing text file which contains all the sentences that you want to test.
* Run [Testing.py](Testing.py) file.
## Result
Evaluation meaning to measure the performance of trained model on the benchmark. In our experiment, we used testing set that we created before training to evaluate the model. We evaluated our model using different parameters: recall, precision, accuracy, F1 score, and confusion matrix.

| Matrix | Result |
| ---- | ---- |
| Precision | 0.80 |
| Recall | 0.80 |
| F1 score | 0.79 |
| Accuracy | 79.74 % |
