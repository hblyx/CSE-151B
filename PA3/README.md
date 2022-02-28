[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7000848&assignment_repo_type=AssignmentRepo)
# cse151b-wi22-pa3

data.py, engine.py, and model.py are required to run the PA3 notebooks. Additionally, use get_data.sh to get data with prepare_data.py.

Specifically, data.py contains methods handling data and data argumentationd, model.py is codes which construct the models. engine.py contains codes for training and experiments but most experiments are implemented in the notebooks.

main.py not implemented since we ran all our experiments and training through notebooks with DataHub and GPU assigned.

PA3_baseline.ipynb contains the training process and visualize the test/validation performance, loss plot, and accuracy plot of the baseline model.

PA3_custom.ipynb contains the training process and visualize the test/validation performance, loss plot, and accuracy plot of the custom model. Experiments of custom model cannot be fully displayed in the notebook, since we did use multiple notebook and additional PC with GPU to do the experiment. We can only show the final (combined) custom model. Specifically, since we did not really record logs of training and experiment, we can not reproduce all experiments. However, since we recorded performance data by hand, we can also make the Change Table for experiments.

PA3_vgg.ipynb contains the training process and visualize the test/validation performance, loss plot, and accuracy plot of the custom model. Experiments of vgg16 model cannot be fully displayed in the notebook, since we directly change codes in the notebook and not keeping all code changes. The data argumentation for the train set are included in the notebook instead of data.py.

PA3_resnet.ipynb contains the training process and visualize the test/validation performance, loss plot, and accuracy plot of the custom model. The data argumentation for the train set are included in the notebook instead of data.py.

Feature Maps.ipynb contains the code for generating weights maps and feature maps for each model.

We do not include checkpoints for models since they are too large.