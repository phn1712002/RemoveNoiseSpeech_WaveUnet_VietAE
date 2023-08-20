# Environment Variables
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'
PATH_LOGS = './Checkpoint/logs/'
PATH_TENSORBOARD = './Checkpoint/tensorboard/'
PATH_TFLITE = './Checkpoint/export/'

# Argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_config', type=bool, default=False, help='Pretrain model WaveUnet in logs training in dataset')
parser.add_argument('--path_file_pretrain', type=str, default='', help='Path file pretrain model')
parser.add_argument('--export_tflite', type=bool, default=False, help='Export to tflite')
args = parser.parse_args()


# Get config
from Tools.Json import loadJson
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_model', 'config_opt', 'config_other', 'config_train', 'config_dataset']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_model = config['config_model']
        config_opt = config['config_opt']
        config_other = config['config_other']
        config_train = config['config_train']
        config_dataset = config['config_dataset']
    else:
        raise RuntimeError("Error config")
            
# Turn off warning
import warnings
if not config_other['warning']:
    warnings.filterwarnings('ignore')
                
# Import dataset
from Dataset.Createdataset import DatasetWaveUnet
train_dataset_raw, dev_dataset_raw, test_dataset_raw = DatasetWaveUnet(path=PATH_DATASET)()
           
# Create optimizers
from Optimizers.OptimizersWaveUnet import CustomOptimizers
opt_waveunet = CustomOptimizers(**config_opt)()

# Create Pipeline dataset
from Architecture.Pipeline import PipelineWaveUnet
pipeline = PipelineWaveUnet(config_model=config_model)

train_dataset = PipelineWaveUnet(params_noise=config_dataset['params_noise'], 
                                 config_model=config_model)(dataset=train_dataset_raw, batch_size=config_dataset['batch_size_train'],)

dev_dataset = PipelineWaveUnet(params_noise=config_dataset['params_noise'], 
                               config_model=config_model)(dataset=dev_dataset_raw, batch_size=config_dataset['batch_size_dev'], )

# Callbacks
from Tools.Callbacks import CreateCallbacks
callbacks_WaveUnet = CreateCallbacks(PATH_TENSORBOARD=PATH_TENSORBOARD, 
                                PATH_LOGS=PATH_LOGS, 
                                config=config, 
                                train_dataset=train_dataset, 
                                dev_dataset=dev_dataset, 
                                pipeline=pipeline)
 

# Create model
from Architecture.Model import WaveUnet
remove_noise = WaveUnet(**config_model, opt=opt_waveunet).build(config_other['summary']) 
   
# Pretrain
from Tools.Weights import loadNearest, loadWeights
if args.pretrain_config:
    if args.path_file_pretrain == '':
        remove_noise = loadNearest(class_model=remove_noise, path_folder_logs=PATH_LOGS)
    else: 
        remove_noise = loadWeights(class_model=remove_noise, path=args.path_file_pretrain)

# Train model
remove_noise.fit(train_dataset=train_dataset,
                 dev_dataset=dev_dataset,
                 epochs=config_train['epochs'],
                 callbacks=callbacks_WaveUnet)
        
# Export to tflite
from Tools.TFLite import convertModelKerasToTflite
if args.export_tflite:
    convertModelKerasToTflite(class_model=remove_noise, path=PATH_TFLITE)

# Off Wandb
import wandb
wandb.finish()