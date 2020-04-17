import yaml
import os

class Config:
    """
    Useful for configuration
    """
    def __init__(self):
        with open(os.getcwd()+"/config/app.yaml","r") as file:
            cfg = yaml.load(file)
        self.INPUT_DIR = cfg['dataset']['input_dir']
        self.INTERIM_DIR = cfg['dataset']['interim_dir']
        self.TRAIN_FILE = cfg['dataset']['train_file']
        self.TEST_FILE = cfg['dataset']['test_file']
        self.RAW_LABEL = cfg['labels']
        self.SEED = cfg['seed']
        self.locked = 1 # this will create lock for adding constants 
    

