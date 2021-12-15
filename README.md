# project_machine_learning
This project is implemented in Python.

The objective of this project is to develop a code allowing the binary classification onto two different datasets : Banknote Authentication Dataset and Chronic Kidney
Disease Dataset.

We can find a repository called models in which we implemented several classification methods: SVM, Neural Network and Decision tree as well as Ada Boost and Random Forest.

The repository test aims to test all implemented functions(unitary tests using pytest) .

The file utils contains the class CleanData which is very important to format the datasets before applying all methods.

Finally a jupyter Notebook called main presents the results on both databases. 

However, Decision Tree is not working well on the Chronic Kidney Disease Dataset since it gives us a model with a maximum depth equal to 1. 
