import wandb 
import tensorflow as tf
import numpy as np
from keras.callbacks import Callback
from Tools.Weights import getPathWeightsNearest
from Architecture.Pipeline import PipelineWaveUnet

class CustomCallbacksWandB(Callback):
    def __init__(self, pipeline: PipelineWaveUnet, path_logs='./Checkpoint/logs/', dev_dataset=None,):
        super().__init__()
        self.dev_dataset = dev_dataset
        self.pipeline = pipeline
        self.path_logs = path_logs
        self.__last_name_update = None
        
    def on_epoch_end(self, epoch: int, logs=None):

        # Sao lưu một mẫu âm thanh kiểm tra
        tableOutputPredict = wandb.Table(columns=["Epoch", "Input", "Speech"])
        for X, _ in self.dev_dataset.take(1):
            if not X.shape[0] == 1:
                index = np.random.randint(low=0, high=X.shape[0] - 1)
                X = X[index]
                X = tf.expand_dims(X, axis=0)
                
        Y = self.pipeline.predictInCallbacks(self.model, X)
        audio_input_wandb = wandb.Audio(tf.squeeze(X).numpy(), 
                                        sample_rate=self.pipeline.sr)
        
        audio_output_wandb = wandb.Audio(Y.numpy(), sample_rate=self.pipeline.sr)
            
        tableOutputPredict.add_data(epoch + 1, audio_input_wandb, audio_output_wandb)
        wandb.log({'Predict': tableOutputPredict}) 
       
        # Cập nhật file weights model to cloud wandb
        path_file_update = getPathWeightsNearest(self.path_logs)
        if self.__last_name_update != path_file_update: 
            self.__last_name_update = path_file_update
            wandb.save(path_file_update)
        