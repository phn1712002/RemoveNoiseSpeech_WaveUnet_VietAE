# Environment Variables
PATH_CONFIG = './config.json'
PATH_LOGS = './Checkpoint/logs/'
PATH_TFLITE = './Checkpoint/export/'

# Argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_file', type=str, default='', help='Path file weights model')
args = parser.parse_args()

# Get config
from Tools.Json import loadJson
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_model', 'config_other']
    if all(key in config for key in keys_to_check):
        config_model = config['config_model']
        config_other = config['config_other']
    else:
        raise RuntimeError("Error config")

# Create model
from Architecture.Model import WaveUnet
remove_noise = WaveUnet(**config_model).build(config_other['summary']) 

# Pretrain
from Tools.Weights import loadNearest, loadWeights
if args.path_file == '':
    remove_noise = loadNearest(class_model=remove_noise, path_folder_logs=PATH_LOGS)
else: 
    remove_noise = loadWeights(class_model=remove_noise, path=args.path_file)

# Export to tflite
from Tools.TFLite import convertModelKerasToTflite
convertModelKerasToTflite(class_model=remove_noise, path=PATH_TFLITE)