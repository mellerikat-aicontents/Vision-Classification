import argparse
import time
import os
os.chdir(os.path.abspath(os.path.join('./alo')))
from src.alo import ALO
from src.alo import AssetStructure
from src.external import external_load_data, external_save_artifacts
import pickle
from glob import glob
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

EVAL_PATH = './evaluation_data'
PIPELINE_DICT = {'train_pipeline':'train', 'inference_pipeline':'inference'}
CARDINALITY_LIMIT = 20

class Wrapper(ALO):
    def __init__(self, nth_pipeline, eval_report=True):
        super().__init__()
        self.preset()
        pipelines = list(self.asset_source.keys())
        self.pipeline = pipelines[nth_pipeline]
        self.eval_report = eval_report
        
        external_load_data(pipelines[nth_pipeline], self.external_path, self.external_path_permission, self.control['get_external_data'])
        self.install_steps(self.pipeline, self.control["get_asset_source"])
        envs, args, data, config = {}, {}, {}, {}
        self.asset_structure = AssetStructure(envs, args, data, config)
        self.set_proc_logger()
        self.step = 0
        self.args_checker = 0
        
        if eval_report:
            os.makedirs(EVAL_PATH, exist_ok=True)
        
        
    def get_args(self, step=None):
        if step is None:
            step = self.step
        self.args = super().get_args(self.pipeline, step)
        self.asset_structure.args = self.args
        self.args_checker = 1

        return self.args
    
    def run(self, step=None, args=None, data=None):
        if self.args_checker==0:
            self.get_args()
        
        if step is None:
            step = self.step
        if args is not None:
            self.asset_structure.args = args
        if data is not None:
            self.asset_structure.data = data
            
        self.asset_structure = self.process_asset_step(self.asset_source[self.pipeline][step], step, self.pipeline, self.asset_structure)
        
        self.data = self.asset_structure.data
        self.args = self.asset_structure.args
        self.config = self.asset_structure.config
        
        if self.eval_report:
            self.save_pkl(self)
        
        self.step += 1
        self.args_checker = 0
        
    def save_pkl(self, obj):
        path = '{eval_path}/{pipeline}_{step}.pkl'.format(
            eval_path=EVAL_PATH, 
            pipeline=PIPELINE_DICT[self.pipeline],
            step=self.step)
        with open(path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
    def load_pkl(self, pipeline, step):
        path = '{eval_path}/{pipeline}_{step}.pkl'.format(
            eval_path=EVAL_PATH, 
            pipeline=pipeline,
            step=step)
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return obj
