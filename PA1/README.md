
CSE 151B PA 1

data.py, network.py, and image.py are required to run the PA1_logreg.ipynb

Specifically, data.py contains methods handling data, network.py is the body of the logistic regression and softmax regression, and image.py is used to generate images from arrays to visualize data and weights.

main.py contains cross_validation_train() which train and validate the model.

We implemented the loading data, simple exploratary analysis of dataset, training, and validation in PA1_logreg.ipynb instead of using main.py. To reproduce the result, put all files together, and run all cells in the PA1_logreg.ipynb. Although we set the random seed of numpy, there might still somethings not reproducible due to the randomness, especially the tuning process.
