"""
Authors : 
    Davide Galli (davide.galli@polimi.it),
    Francesco Lattari,
    Matteo Matteucci (matteo.matteucci@polimi.it),
    Davide Zoni (davide.zoni@polimi.it)
"""

import os
import shutil
from omegaconf import OmegaConf

from datetime import datetime

import neptune.new as neptune

from pytorch_lightning.loggers import NeptuneLogger


def build_neptune_logger(exp_name, tags, neptune_config_file):
    neptune_config = OmegaConf.to_object(
        OmegaConf.load(neptune_config_file))
    
    user = neptune_config['user']
    token = neptune_config['token']
    project = neptune_config['project']
    id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    kwargs = {}
    kwargs['prefix'] = 'experiment'
    kwargs['project'] = f'{user}/{project}' 
    kwargs['name'] = f'{exp_name}_{id}' 
    kwargs['description'] = ''
    kwargs['tags'] = tags 
    
    kwargs['source_files'] = [
        '/CNN/*'
    ]
    
    neptune_logger = NeptuneLogger(api_key=token, **kwargs)
    return neptune_logger, kwargs['name']

def get_neptune_run(neptune_config_file, SID):
    neptune_config = OmegaConf.to_object(
        OmegaConf.load(neptune_config_file))
    
    user = neptune_config['user']
    token = neptune_config['token']
    project = neptune_config['project']

    kwargs = {}
    kwargs['prefix'] = 'experiment'
    kwargs['project'] = f'{user}/{project}' 
    
    project = neptune.init_project(
        api_token=token, project=f'{user}/{project}', mode='read-only') # with_id=SID) 
    
    # project = neptune.init_project(
    #     api_token=token, name=f'{user}/{project}', mode='read-only') 
    
    df = project.fetch_runs_table().to_pandas()
    # print(df.columns)

    # print(os.path.basename(df['experiment/model/best_model_path'][0]))

    # print("\nNeptune run with id {} reloaded.".format(
    #     run['sys/id'].fetch()))
    # print("Experiment name: {}\n".format(
    #     run['sys/name'].fetch()))
    df = df[df['sys/id'] == SID]
        
    return df 

def init_experiment_dir(exp_config, exp_name):
    exp_dir = os.path.join(exp_config['log_dir'], exp_name)
    print(exp_dir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    else:
        raise Exception("Experiment folder already exists.")
    return exp_dir

def save_experiment_configs(exp_dir, original_config_dir):
    config_dir = os.path.basename(os.path.normpath(original_config_dir))
    shutil.copytree(
        original_config_dir, 
        os.path.join(exp_dir, 'configs', config_dir))