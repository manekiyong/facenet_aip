from model.inception_resnet_v1 import InceptionResnetV1
from model.mtcnn import MTCNN

import torch
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
from clearml import Task, Dataset


PROJECT_NAME = 'facenet'

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


class Generate(object):

   # should init as arguments here
    def __init__(self, args):
        if args.clearml:
            # self.clearml_task = Task.get_task(project_name=PROJECT_NAME, task_name='pl_generate')
            self.clearml_task = Task.init(project_name=PROJECT_NAME, task_name='pl_generate_'+args.exp_name) # DEBUG
            # self.clearml_task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04", 
                # docker_setup_bash_script=['pip3 install torchvision']
            # )
            # self.clearml_task.execute_remotely(queue_name="compute")
        self.s3 = args.s3
        self.input = os.path.join(args.input, '')
        self.resnet = InceptionResnetV1(pretrained=None, classify=False)
        self.model_path = args.model_path
        self.output = os.path.join(args.output, '')
        if self.s3:
            # if args.clearml: # DEBUG
            #     self.clearml_task.execute_remotely(queue_name="compute") # DEBUG
            dataset_name = args.s3_dataset_name
            dataset_project = "datasets/facenet"
            # Get image Dataset
            s3_dataset_path = Dataset.get(
                dataset_name=dataset_name, 
                dataset_project=dataset_project
            ).get_local_copy()
            s3_dataset_path = os.path.join(s3_dataset_path, '')
            self.input=s3_dataset_path+self.input
            # Get Trained Model
            s3_model_path = Dataset.get(
                dataset_name=args.exp_name+'_models', 
                dataset_project = 'datasets/facenet'
            ).get_local_copy()
            s3_model_path= os.path.join(s3_model_path, '')
            self.model_path = s3_model_path+self.model_path
            # Create Dataset for the generated embeddings
            self.emb_dataset = Dataset.create(
                dataset_name=args.exp_name+'_embeddings', 
                dataset_project = 'datasets/facenet'
            )

        self.resnet.load_state_dict(torch.load(self.model_path), strict=False)
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
        if self.s3:
            self.emb_dataset.add_files(emb_folder+emb_id+'.pt', dataset_path=emb_folder)
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
        if self.s3:
            self.emb_dataset.upload(output_url='s3://experiment-logging/')
            self.emb_dataset.finalize()
            self.emb_dataset.publish()
            



    @staticmethod
    def add_generate_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exp_name",
            default='experiment',
            help="Experiment Name"
        )
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
        parser.add_argument(
            "-c",
            "--clearml",
            action="store_true",
            help="Connect to ClearML"
        )        
        parser.add_argument(
            "-s",
            "--s3",
            action="store_true",
            help="Call to use s3"
        )
        parser.add_argument(
            "--s3_dataset_name",
            default='vggface_exp10',
            help="ClearML Dataset Name"
        )

        return parser

if __name__ == '__main__':
    parser = Generate.add_generate_args()
    args = parser.parse_args()
    gen = Generate(args)
    gen.generate_all()