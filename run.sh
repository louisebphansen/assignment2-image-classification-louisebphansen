source env/bin/activate # activate virtual environment

# run scripts
python3 src/logistic_classification.py --penalty l1
python3 src/nn_classification.py --hidden_layer_sizes 64 10 --early_stopping True

deactivate # deactivate virtual environment again