import argparse
import time
import os
if os.path.abspath('.').split('/')[-1]!='alo':
    os.chdir(os.path.abspath(os.path.join('./alo')))
from src.alo import ALO
from src.alo import AssetStructure
from src.external import external_load_data, external_save_artifacts
import pickle
from glob import glob
import_libraries = False
try:
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    import pandas as pd
    import matplotlib.pyplot as plt
    import missingno as msno
    import tensorflow as tf
    from PIL import Image
    import numpy as np
    import copy
    import_libraries = True
except:
    pass

EVAL_PATH = './evaluation_data'
PIPELINE_DICT = {'train_pipeline':'train', 'inference_pipeline':'inference'}
CARDINALITY_LIMIT = 20

class Wrapper(ALO):
    def __init__(self, nth_pipeline, exp_plan_file='experimental_plan.yaml', eval_report=True):
        global import_libraries
        super().__init__()
        self.exp_plan_file = exp_plan_file

        self.set_proc_logger()
        self.preset()
        self.set_asset_structure()
        if not import_libraries:
            import seaborn as sns
            from sklearn.metrics import classification_report, confusion_matrix
            import pandas as pd
            import matplotlib.pyplot as plt
            import missingno as msno
            import tensorflow as tf
            from PIL import Image
            import numpy as np
            import copy
            import_libraries = True

        pipelines = list(self.asset_source.keys())
        self.pipeline = pipelines[nth_pipeline]
        self.eval_report = eval_report
        
        external_load_data(pipelines[nth_pipeline], self.external_path, self.external_path_permission, self.control['get_external_data'])
        self.install_steps(self.pipeline, self.control["get_asset_source"])
        
        self.set_proc_logger()
        self.step = 0
        self.args_checker = 0
        
        if eval_report:
            os.makedirs(EVAL_PATH, exist_ok=True)
        
        
    def get_args(self, step=None):
        if step is not None:
            self.step = step
        self.args = super().get_args(self.pipeline, self.step)
        self.asset_structure.args = self.args
        self.args_checker = 1

        return self.args
    
    def run(self, step=None, args=None, data=None):
        if self.args_checker==0:
            self.get_args()
        
        if step is not None:
            self.step = step
        if args is not None:
            self.asset_structure.args = args
        if data is not None:
            self.asset_structure.data = data
        
        self.asset_structure = self.process_asset_step(self.asset_source[self.pipeline][self.step], self.step, self.pipeline, self.asset_structure)
        
        self.data = self.asset_structure.data
        self.args = self.asset_structure.args
        self.config = self.asset_structure.config
        
        if self.eval_report:
            self.save_pkl(self)
        
        self.step += 1
        self.args_checker = 0
        
    def save_pkl(self, obj):

        if self.asset_source[self.pipeline][self.step]['step'] == 'train':
            tf.data.Dataset.save(self.data['original_image_dataset'],'{eval_path}/{pipeline}_{step}_original_image_dataset'.format(
            eval_path=EVAL_PATH, 
            pipeline=PIPELINE_DICT[self.pipeline],
            step=self.step))
            tf.data.Dataset.save(self.data['augmented_image_dataset'],'{eval_path}/{pipeline}_{step}_augmented_image_dataset'.format(
            eval_path=EVAL_PATH, 
            pipeline=PIPELINE_DICT[self.pipeline],
            step=self.step))

        self.data.pop('original_image_dataset',None)
        self.data.pop('augmented_image_dataset',None)

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

        if self.asset_source[self.pipeline][self.step]['step'] == 'inference':
            loaded_ori_dataset = tf.data.Dataset.load('{eval_path}/{pipeline}_{step}_original_image_dataset')
            loaded_aug_dataset= tf.data.Dataset.load('{eval_path}/{pipeline}_{step}_augmented_image_dataset')

            obj.data['original_image_dataset'] = loaded_ori_dataset
            obj.data['augmented_image_dataset'] = loaded_aug_dataset
        
        return obj


    def plot_training_history(self):
        try:
            training_history = self.asset_structure.data['eval_score']['training_arguments']['history']
        except:
            raise Exception('failed loadinig history')

        trn_loss = training_history['loss']
        val_loss = training_history['val_loss']
        trn_acc = training_history['accuracy']
        val_acc = training_history['val_accuracy']

        fig, axes = plt.subplots(nrows=1, ncols=2, constrained_layout=True,figsize=(8,3))

        axes[0].set_title('Model Loss')
        axes[0].plot(range(len(trn_loss)),trn_loss,label='training_loss')
        axes[0].plot(range(len(val_loss)),val_loss,label='validation_loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].set_title('Model Accuracy')
        axes[1].plot(range(len(trn_acc)),trn_acc,label='training_accuracy')
        axes[1].plot(range(len(val_acc)),val_acc,label='validation_accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.show()   


def make_ground_truth(path,save_path,data_type='png'):

    path_list = []
    col_names = ['label','image_path']
    
    data_type_dict = {}
    data_type_dict['png'] = ['png']
    data_type_dict['jpg'] = ['jpg']
    data_type_dict['both'] =  ['png','jpg']
    
    
    WALK = os.walk(path)
    i = 0
    for (Root,Dir,fles) in WALK:
        for fle in fles:
            if fle.split('.')[-1] in data_type_dict[data_type]:
                path_list.append([Root.split('/')[-1] , os.path.join(Root,fle)])
                if i<5:
                    print(path_list[-1])
                    i += 1
            
    df = pd.DataFrame(path_list,columns = col_names)
    df.to_csv(save_path,index=False)
    
    return path_list,df




def plot_auged_images(img_path_lst,resize_shape,aug_lst=None):

    from alo.assets.train.augment import RandAugment
    
    available_aug_lst = [
        'AutoContrast', 'Equalize', 'Invert', 'Rotate', 'Posterize', 'Solarize',
        'Color', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY',
        'TranslateX', 'TranslateY', 'Cutout', 'SolarizeAdd'
    ]
    
    if aug_lst == None:
        aug_lst = copy.deepcopy(available_aug_lst)
    
    def load_image(img_path, target_size):

        if img_path.endswith('npy'):
            image = np.load(img_path)
        else:
             image = Image.open(img_path)
        if image is not None:
            image = np.array(image)
            
        else:
            raise Exception('image open failed')

        if len(image.shape) != 3:
            image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, target_size)  # 이미지 리사이징
        if image.shape[2] != 3:
            image = tf.tile(image, [1, 1, 3])
        return image
    
    total_image_result = [[] for i in range(len(img_path_lst))]
    
    for img_idx , each_img_path in enumerate(img_path_lst):
        
        original_image = load_image(img_path=each_img_path,target_size=resize_shape[:2])
        total_image_result[img_idx].append(original_image)
        for each_aug in aug_lst:
            
            randaugmenter = RandAugment(num_layers=1, exclude_ops = [i for i in available_aug_lst if i != each_aug])
            auged_image = randaugmenter.distort(original_image)
            
            total_image_result[img_idx].append(auged_image)
    
    aug_lst.insert(0,'Original')
    
    fig, axes = plt.subplots(nrows=len(img_path_lst) , ncols=len(aug_lst),constrained_layout=True)
    for each_image_idx in range(len(img_path_lst)):
        for each_aug_idx in range(len(aug_lst)):
            if each_image_idx == 0:
                axes[each_image_idx][each_aug_idx].set_title(aug_lst[each_aug_idx],fontsize=7)
            axes[each_image_idx][each_aug_idx].imshow(total_image_result[each_image_idx][each_aug_idx]/255)
            axes[each_image_idx][each_aug_idx].axis('off')
                
    plt.show()
