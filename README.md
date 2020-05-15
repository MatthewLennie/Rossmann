# Rossman Tabular NN
The aim of this project is to solve the rossman kaggle challenge as a practice for making neater and better functioning code. 
The code aims to implement the best practices that I am aware of from the FASTAI courses, these include cyclical learning rates, batch norm, good data normalization practices. 
I also used a "Jeremy" rule of thumb for the embedding sizes. 

## About the Code
Rossman_tabular contains the main running code. 
Import_rossman_data contains the data-preprocessing. These should be run seperately. 
As described in the Import_rossman_data file, I did not do much with the data and essentially took a cleaned data set. 
Data Munging was not what I was aiming to practice here. 

Visualisation is provided through tensorboard, this can be launched with: tensorboard --logdir runs
activations can added to the tensorboard with learner.model.put_activations_into_tensorboard=True

## Things learnt on this project:
- Default momentum was pretty good, But did manage a better result with lower momentum
- Go easy on the dropout. I tried 0.8 just out of curiosity, destabilzes the training pretty severely.
- Make sure you delete old models to ensure that the GPU does garbage collection. 
- watch gpu resources with: watch -d -n 0.5 nvidia-smi 
- htop for watching resources
- train from TMUX on server so that ssh dropouts doesn't interrupt training
- Make sure to put model into eval mode when you have dropout to avoid using dropout on validation set. 

## Things to improve in next project:
- Better data visualization
- LRFinder
- visualize gradients. 
- Possible clip gradients to avoid exploding training scenarios. Wasn't able to confirm this hypothesis.
- WAAAAY better error handling in training.  
- saving and recovery of models. 
- Better experimental logging. System here wasn't really satisfying. 
- Better initialization, the current automatic intitialization seems to work well enough but could be better
- Doing checks on tensors during training is inefficient and can create copies of tensors. 

## Best Result and Hyperparameters
Validation Error: 5e-5 MSE (note I am not comparing to the Percentage MSE)
- Cosine:10, 
- lr: 0.0005, 
- batchsize: 100000, 
- layers:[248, 240, 150, 80, 40, 10, 2], 
- dropout:0.3, 
- momentum: [0.999, 0.99999]

