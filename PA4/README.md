[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7169855&assignment_repo_type=AssignmentRepo)
Change this README to describe how your code works and how to run it before you submit it on gradescope.

# Image Captioning

* Yaoxin Li: implemented deterministic samples generating approach, implemented traning and validation in the notebooks, implemented baseline model, RNN model, and tuned LSTM model.
* Mengxuan Li: implemented stochastic samples generating approach, generate examples for models
* Xingyu Zhu: implemented Architecture2 model, implemented training and evaluating for Architecture2 model
* Yiming Hao:

## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
  * Instead of implementing `experiment.py`, we implemented `baseline.ipynb`, `LSTM.ipynb`, `RNN.ipynb`, `arch2.ipynb` to do the equivalent functions in `experiment.py` for each model. 
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
  * Since we acutally totally use notebooks to do experiments, we did not use `main.py`. All configurations can be found in the notebooks.
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
  * We stored our examples and loss plots in a separate folder. However, as the examples and plots are too much, we might not upload it with codes. But we have already includes them in the notebook and the report.
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.
  * We saved all models in checkpoints. However, since checkpoints are extremely large files, we might not upload them with codes. Thus, to resume or re-perform an experiment, we loaded the mdoels from the checkpoints to do so. 
## Files
- `main.py`: Main driver class
  - We did not implemented the `main.py` since we did all experiments with notebooks.
- `experiment.py`: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
  - We did not implemented the `experiment.py` since we did all experiments with notebooks. 
  - We actually directly implemented the training, evaluate, generating samples in the notebooks.
- `dataset_factory`: Factory to build datasets based on config
  - We slightly modified the `dataset_factory.py` to enable us to attempt some data augmentation
- `model_factory.py`: Factory to build models based on config
  - We build all of our models in the model_factory.py
- `constants.py`: constants used across the project
  - We only store the contents of the baseline model into the `constants.py`
  - Since the DataHub Limit, we have to limit our batch size to 20. 
- `file_utils.py`: utility functions for handling files
- `caption_utils.py`: utility functions to generate bleu scores
- `vocab.py`: A simple Vocabulary wrapper
- `coco_dataset.py`: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
  - we slightly modified the `coco_dataset.py` to enable us to attempt some data augmentation
- `get_datasets.ipynb`: A helper notebook to set up the dataset in your workspace
- `baseline.ipynb`: notebook contains all codes for training, evaluating, and generating samples for the baseline model
- `LSTM.ipynb`: notebook contains all codes for training, evaluating, and generating samples for the tuned LSTM model
- `RNN.ipynb`: notebook contains all codes for training, evaluating, and generating samples for the RNN model
- `arch2.ipynb`: notebook contains all codes for training, evaluating, and generating samples for the Architecture2 model
- `savedVocab`: vocabularies we used to train and generate examples for all models
  - since we use different vocabulary threshold for the models, we also include this file