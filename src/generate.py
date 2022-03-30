
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# from pytorch_lightning.loggers import TensorBoardLogger
# from aiplatform.s3utility import S3Callback, S3Utils
# from aiplatform.config import cfg as aip_cfg
# from .model import  
# from . import transforms
# from .config import cfg
# from .dataset import FlightDataset


# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from model.inception_resnet_v1 import InceptionResnetV1
from model.mtcnn import MTCNN, fixed_image_standardization
from model.utils import training
from data.preprocessor import PreProcessor

import torch
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
from clearml import Task


PROJECT_NAME = 'facenet'

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


class Generate(object):

   # should init as arguments here
    def __init__(self, args):
        # self.clearml_task = Task.get_task(project_name=PROJECT_NAME, task_name='pl_generate')
        self.input = os.path.join(args.input, '')
        self.output = os.path.join(args.output, '')
        self.resnet = InceptionResnetV1(pretrained=None, classify=False)
        self.resnet.load_state_dict(torch.load(args.model_path), strict=False)
        self.resnet.eval()
        self.mtcnn = MTCNN(image_size=160, margin=0, device='cuda', keep_all=True)


    def generate_embedding(self, emb_id, img_folder='train/', emb_folder='emb/'):
        # # Generate Embedding, and copy holdout photos to eval folder    
        folder_path = img_folder+'/'+emb_id+'/'
        folder_content = os.listdir(folder_path)

        avg_emb = torch.zeros(512) # Reset Avg Embedding
        img_count = 0 # Reset Image Count
        emb_count = 0
        
        for i in folder_content:
            img_count +=1
            img = Image.open(folder_path+i)
            img_cropped, prob = self.mtcnn(img, return_prob=True)
            if prob[0] != None:
                emb_count+=1
                max_val = np.argmax(prob)
                img_embedding = self.resnet(img_cropped)
                avg_emb = avg_emb.add(img_embedding[max_val])

        avg_emb=avg_emb.div(emb_count)
        torch.save(avg_emb, emb_folder+emb_id+'.pt')
        print(emb_id, "Done")
        return


    def generate_all(self):
        count=0
        Path(self.output).mkdir(parents=True, exist_ok=True)
        for i in os.listdir(self.input):
            count+=1
            self.generate_embedding(i, 
                        self.input,
                        emb_folder=self.output)
            



    @staticmethod
    def add_generate_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i",
            "--input",
            default='train/',
            help="Data Set image folder path"
        )
        parser.add_argument(
            "-o",
            "--output",
            default='emb/',
            help="Embedding output folder"
        )
        parser.add_argument(
            "-m",
            "--model_path",
            default='model.pt',
            help="Path & Model Name"
        )

        return parser

if __name__ == '__main__':
    parser = Generate.add_generate_args()
    args = parser.parse_args()
    gen = Generate(args)
    gen.generate_all()