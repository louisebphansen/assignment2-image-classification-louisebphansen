[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10449089&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

This repository contains the code for Assignment 2 in the Visual Analytics course on the Cultural Data Science elective. 

### Contributions
The code was created by me, but code from the notebooks provided throughout the course has been used and adapted.

### Assignment description
For this assignment, we'll be writing scripts which classify the Cifar10 dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier and one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via scikit-learn.

### Methods and contents

#### Contents
| Folder/File  | Contents| Description |
| :---:   | :---: | :--- |
|```out```|log_clf_report, nn_clf_report.txt| The folder contains the classification reports from running the logisitic classification and Neural Network classification scripts. |
|```src```|data.py, logistic_classification.py, nn_classification.py| The folder contains scripts to preprocess data (**data.py**), run a logistic classifier (**logistic_classification.py**) and a neural network classifier (**nn_classification.py**).|
|README.md|-| Description of repository and how to run the code|
|requirements.txt|-|Packages required to run the code|
|run.sh|-|Bash script for running logistic classifier and neural network classifier with predefined arguments|
|setup.sh|-|Bash script for setting up virtual environment and installing packages|

#### Data
This assignment uses the cifar10 dataset, which consists of 50,000 training images and 10,000 test images across 10 different categories. It is already split into training and testing data which can be directly loaded through the Keras wrapper for Tensorflow. 

#### Methods
*The following section describes the methods used in the provided Python scripts.*

**Data preprocessing**

As the dataset consists of color images, all images are converted to greyscale. Next, they are standardized by dividing pixel values with 255. Finally, the data is reshaped to be in the shape (nsamples, xsize, ysize), resulting in preprocessed training and testing datasets ready for a classifier. 


**Logistic Regression Classifier**

A logistic regression classifier is trained using *scikit-learn*. As there are 10 outcome classes, it is a Multinomial Logistic Regression. The code is designed to allow for different methods of solvers and regularization penalties, depending on what arguments is passed (see **Usage**). The results from the classifier can be seen in the ```out``` folder.


**Neural Network classifier**

A neural network is trained using the *MLP (Multilayer Linear Perceptron)* class from *scikit-learn*. The MLP classifier creates a fully connected feed-forward neural network with hidden layers. The final layer consists of a classification, predicting the probability of each of the 10 output classes. Like the logistic classifier, the code is designed to allow for different hyperparameters, depending on the arguments passed (see **Usage**). The script also saves a classification report in the ```out``` folder. 


### Usage

All code for this project was designed to run on an *Ubuntu 22.10* operating system. 

To reproduce the results in this repository, clone this repository using ```git clone```.

It is important that you run all scripts from the *assignment2-image-classification-louisebphansen* folder, i.e., your terminal should look like this:

```
--your_path-- % assignment2-image-classification-louisebphansen %
```

##### Setup 
First, ensure that you have installed the **venv** package for Python (if not, run ```sudo apt-get update``` and ```sudo apt-get install python3-venv```). 

To set up the virtual environment, run ```bash setup.sh``` from the terminal.

##### Run code
To run the code, you can do the following:

##### Run both Python scripts with predefined arguments
To run both the Logistic Classification script and the Neural Network Classification scripts with pre-defined arguments, type ```bash run.sh``` in the terminal. The logistic regression is run using the default values for tolerance for stopping criteria (*0.1*), solver (*saga*) and name of classification report '*log_clf_report.txt*'. The regularization penalty is defined as *l1*. The Neural Network script is run with two hidden layers of sizes 64 and 10, early stopping set to true, and default values for learning rate (*adaptive*), maximum iterations (*15*) and name of the classification report (*'nn_clf_report.txt'*). See below for a further explanation of arguments. The results from this run can be seen in the ```out```folder and in the **Results** section. 



##### Define arguments yourself
Alternatively, you can run the scripts separately or define the arguments yourself. Again, it is important that you run it from the main folder. From the terminal, first activate the virtual environment, then run the script(s) with the desired arguments:

```
source env/bin/activate

python3 src/logistic_classification.py --penalty <penalty> --tol <tol> -- solver <solver> --clf_report_name <clf_report_name> 

``` 
**Arguments**

- **penalty:** Regularization penalty. Default: 'None'
- **tol:** Tolerance for stopping criteria. Default: 0.1
- **solver:** Optimization algorithm. Default: 'saga'
- **clf_report_name:** Name of the output classification report. Default: 'log_clf_report.txt'

```
python3 src/nn_classification.py --hidden_layer_sizes <size> <size> <etc..> --learning_rate <learning_rate> --early_stopping <early_stopping> --max_iter <max_iter> --clf_report_name <clf_report_name>
```

**Arguments**
- **hidden_layer_sizes:** Specify sizes of hidden layer (e.g, *64 10* would be two layers of sizes 64 and 10). No default
- **learning_rate:** Learning rate schedule for how to update the weights. Can be 'constant', 'invscaling' or 'adaptive'. Default: 'adaptive'. 
- **early_stopping:** Whether to stop if validation score is not improving. Boolean. Default: False
- **max_iter:** Maximum number of iterations. Default: 15
- **clf_report_name:** Name of the output classification report. Default: 'nn_clf_report.txt'

### Results

**Logistic Regression Classifier**

![Skærmbillede 2023-05-18 kl  17 05 08](https://github.com/AU-CDS/assignment2-image-classification-louisebphansen/assets/75262659/ec12a4bf-cbd1-4cd3-b8d6-392dd8e36e53)

**Neural Network Classifier**

![Skærmbillede 2023-05-18 kl  17 06 35](https://github.com/AU-CDS/assignment2-image-classification-louisebphansen/assets/75262659/4e8bebb2-abc1-4e32-aaf6-56801c664d58)

Looking at the output from running both the logistic regression and neural network classifiers, it is evident that the neural network classifier is performing a bit better on allmost all categories (except for *airplane*, *cat* and *frog*) when looking at F1 scores. Looking at the overall accuracy, the neural network is also performing better. However, there seems to be some variation in how well the model performs across the different categories. It seems to be better at predicting categories such as *ship*, *truck* and *automobile* than *cat* and *frog*. Even though the neural network performs better than the logistic regression, looking at the cifar-10's Wikipedia page, where accuracies are described to reach above 90%, it is still a fairly bad performing model.
