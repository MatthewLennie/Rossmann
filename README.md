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

HyperParameterSearch sets up a bunch of models and runs them with different hyperparameters. 

## Things learnt on this project:
- Default momentum was pretty good, But did manage a better result with lower momentum
- Go easy on the dropout. I tried 0.8 just out of curiosity, destabilzes the training pretty severely.
- Make sure you delete old models to ensure that the GPU does garbage collection. It's also good to clear the cache. It also seems that entangling the callbacks into the object creates circular references that cause the garbage collection to not work.  
- watch gpu resources with: watch -d -n 0.5 nvidia-smi 
- nvprof is useful for GPU profiling. 
- kernprof is useful for CPU profiling. 
- htop for watching resources
- train from TMUX on server so that ssh dropouts doesn't interrupt training
- Make sure to put model into eval mode when you have dropout to avoid using dropout on validation set. 
- Standard pytorch data loader not optimized for structured data because it fetches each example seperately and then concatenates. This is fine when each sample has megapixel number of features and batch sizes are ~100. When batch sizes are 100k+, this implementation causes the CPU to throttle the GPU by reading the data so slowly. FastTensorDataLoader gives a better approach which achieves much better performance even without paralellization. 
- Avoid writing code in __main__, especially if dealing with pickling data. The pickle files will inherit the context of __main__ which means it will not find the object you are trying to unpickle. Create a separate function. 
## Things to improve in next project:
- Better data visualization
- LRFinder
- visualize gradients. 
- Possible clip gradients to avoid exploding training scenarios. Wasn't able to confirm this hypothesis.
- WAAAAY better error handling in training.  
- saving and recovery of models. 
- Better experimental logging. System here wasn't really satisfying. 
- Better initialization, the current automatic intitialization seems to work well enough but could be better
- Doing checks on tensors (i.e. isnan) during training is inefficient and can create copies of tensors. It would be far more efficient to do it inplace, the whole point is to raise an assertion which can be achieved without creating a binary tensor. 
- Adjustment of batch size for the given model!

## Best Result and Hyperparameters
Validation Error: 0.000094297 MSE (note I am not comparing to the Percentage MSE)
- Cosine:2, 
- lr: 0.001, 
- batchsize: 200000, 
- layers:[248, 60, 60, 40, 30, 20, 10, 2], 
- dropout:0.4,
- momentum: [0.8, 0.99]


