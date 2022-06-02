Very messy and badly commented code but this is the main file for properly running the model on.

**Explanations for what is in the different files**:

**arg_extractor**; is simply the commands for extracting the arguments from running directly on the command line

**combined_loss**; file containing the custom loss function so it doesn't clutter up the main file

**data_generation**; synthetic data generation file, seperated to reduce clutter

**experiment_builder**; old version of the experiment builder file that doesn't contain the ability to run RNN networks

**experiment_builder_rnn**; most recent version of the experiment builder, it can run RNN networks

**main**; I ran single experiment through this file, likely messy due to last minute rush changes

**main_hparam**; ran multiple experiments here, used for hyperparameter testing

**model_architectures**; contains all the models i used

**utils**; misc functions that i needed, like early stopping and learning schedule
