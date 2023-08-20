# Environment variables
PATH_CONFIG = './tuning_hyperparameter.json'
PATH_DATASET = './Dataset/'

# Get config
from Tools.Json import loadJson
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_sweep', 'config_dataset', 'config_other']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_sweep = config['config_sweep']
        config_dataset = config['config_dataset']
        config_other = config['config_other']
    else:
        raise RuntimeError('Error config')

# Create Sweep WandB
import os, wandb
os.environ['WANDB_API_KEY'] = config_wandb['api_key']
wandb.login()
sweep_id = wandb.sweep(config_sweep, project=config_wandb['project'])
        
# Turn off warning
import warnings
if not config_other['warning']:
    warnings.filterwarnings('ignore')
    
# Load dataset
from Dataset.Createdataset import DatasetWaveUnet
train_raw_dataset, dev_raw_dataset, test_raw_dataset = DatasetWaveUnet(path=PATH_DATASET)()

# Split dataset
from Tools.TuningHyper import splitDataset
train_raw_dataset = splitDataset(train_raw_dataset, size_dataset=config_dataset['size_dataset'])
dev_raw_dataset = splitDataset(dev_raw_dataset, size_dataset=config_dataset['size_dataset'])


# Tuning hyperparameter
from Architecture.Model import WaveUnet
from Optimizers.OptimizersWaveUnet import CustomOptimizers
from Architecture.Pipeline import PipelineWaveUnet 
from wandb.keras import WandbCallback
def tuningHyperparamtrer(config=None, params_noise=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Create all config
        config_model = {
            "name": "WaveUnet_Tuning"
            }
        
        config_opt = {
            "learning_rate": pow(10, config.learning_rate)
        }
        
        config_dataset = {
            "batch_size_train": config.batch_size_train,
            "batch_size_dev": config.batch_size_dev,
            "params_noise": params_noise
        }
        
        config_train = {
            "epochs": config.epochs
        }
        
        # Create pipeline 
        train_dataset = PipelineWaveUnet(dataset=train_raw_dataset, 
                                         batch_size=config_dataset['batch_size_train'], 
                                         params_noise=config_dataset['params_noise'], 
                                         config_model=config_model)()
        
        dev_dataset = PipelineWaveUnet(dataset=dev_raw_dataset, 
                                       batch_size=config_dataset['batch_size_dev'], 
                                       params_noise=config_dataset['params_noise'], 
                                       config_model=config_model)()

        # Create optimizers
        opt_waveunet = CustomOptimizers(**config_opt)()

        # Create model 
        remove_noise = WaveUnet(**config_model, opt=opt_waveunet).build() 

        # Train model
        remove_noise.fit(train_dataset=train_dataset, 
                dev_dataset=dev_dataset,
                epochs=config_train['epochs'],
                callbacks=[WandbCallback(log_weights=True, 
                                         log_gradients=True, 
                                         save_model=False, 
                                         training_data=train_dataset,
                                         validation_data=dev_dataset,
                                         log_evaluation=True,
                                         log_batch_frequency=True)])

# Tuning 
wandb.agent(sweep_id, lambda config: tuningHyperparamtrer(config, config_dataset['params_noise']), count=config_wandb['count'])