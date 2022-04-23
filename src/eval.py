from model.inception_resnet_v1 import InceptionResnetV1
from model.mtcnn import MTCNN
from torch.nn.modules.distance import PairwiseDistance

import torch
import numpy as np
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from clearml import Task, Logger, Dataset


PROJECT_NAME = 'facenet'

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


class Evaluate(object):

   # should init as arguments here
    def __init__(self, args):
        self.use_clearml = args.use_clearml
        if self.use_clearml:
            # self.clearml_task = Task.get_task(project_name=PROJECT_NAME, task_name='pl_evaluate')
            self.clearml_task = Task.init(project_name=PROJECT_NAME, task_name='pl_evaluate') # DEBUG
            self.logger = Logger.current_logger()
        self.input = os.path.join(args.input, '')
        self.emb = os.path.join(args.emb, '')
        self.resnet = InceptionResnetV1(pretrained=None, classify=False)
        self.model_path = args.model_path
        self.label = args.label
        self.s3 = args.s3
        if self.s3:
            dataset_name = args.s3_dataset_name
            dataset_project = "datasets/facenet"
            # Get image dataset
            s3_dataset_path = Dataset.get(
                dataset_name=dataset_name, 
                dataset_project=dataset_project
            ).get_local_copy()
            s3_dataset_path = os.path.join(s3_dataset_path, '')
            self.input=s3_dataset_path+self.input
            self.label=s3_dataset_path+self.label
            # Get Trained Model
            s3_model_path = Dataset.get(
                dataset_name=args.exp_name+'_models', 
                dataset_project = 'datasets/facenet'
            ).get_local_copy()
            s3_model_path= os.path.join(s3_model_path, '')
            self.model_path = s3_model_path+self.model_path
            # Get embeddings
            s3_embedding_path = Dataset.get(
                dataset_name=args.exp_name+'_embeddings', 
                dataset_project = 'datasets/facenet'
            ).get_local_copy()
            s3_embedding_path= os.path.join(s3_embedding_path, '')
            self.emb = s3_embedding_path+self.emb
            
        self.resnet.load_state_dict(torch.load(self.model_path), strict=False)
        self.resnet.eval()
        self.mtcnn = MTCNN(image_size=160, margin=0, device='cuda', keep_all=True)
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        
        with open(self.label) as json_file:
            self.golden = json.load(json_file)


    def load_emb(self, folder='emb/', norm=False, transpose=False):
        id_list = []
        for i in os.listdir(folder):
            id_val = int(i[:-3])
            id_list.append(id_val)
        id_list.sort()
        emb_list = []
        for i in id_list:
            temp_emb = torch.load(folder+str(i)+'.pt')
            if norm:
                temp_emb = temp_emb / torch.linalg.norm(temp_emb)
            emb_list.append(temp_emb)
        emb_list = torch.stack(emb_list)
        if transpose:
            emb_list = torch.transpose(emb_list, 0, 1)
        return emb_list, id_list


    def predict_id(self, img_path, emb_list, k=[1,3,5]):
        l2_distance = PairwiseDistance(p=2)
        temp_sim_cos = []
        temp_sim_euc = []
        img = Image.open(img_path)
        img_cropped, prob = self.mtcnn(img, return_prob=True)
        if prob[0] != None:
            max_val = np.argmax(prob)
            img_embedding = self.resnet(img_cropped)
            img_embedding = img_embedding[max_val]
        else:
            return [[-1]]*len(k), [[-1]]*len(k), [[-1]]*len(k), [[-1]]*len(k)
        
        # Compute Cosine Similarity (By Matrix Multiplication)
        temp_sim_cos = torch.matmul(img_embedding, emb_list)
        # Compute L2 Dist Similarity
        t_embedding_list = torch.transpose(emb_list, 0, 1)
        for j in t_embedding_list:
            dist = l2_distance(img_embedding, j).item()
            temp_sim_euc.append(dist)
        temp_sim_euc=np.array(temp_sim_euc)
        # temp_sim_euc=torch.from_numpy(temp_sim_euc)

        indices_list_cos = []
        values_list_cos = []
        indices_list_euc = []
        values_list_euc = []   
        for i in k:
            topk_cos = torch.topk(temp_sim_cos, i)
            topk_euc = np.argpartition(temp_sim_euc, i)[:i]
            topk_euc_conf = temp_sim_euc[topk_euc]
            if set(topk_cos.indices.tolist()) != set(topk_euc):
                print(topk_cos)
                print(topk_euc)
            indices_list_cos.append(topk_cos.indices.tolist())
            values_list_cos.append(topk_cos.values.tolist())
            indices_list_euc.append(topk_euc)
            values_list_euc.append(topk_euc_conf)
        
        return indices_list_cos, values_list_cos, indices_list_euc, values_list_euc

    def evaluate(self):
        embeddings, ids = self.load_emb(self.emb, norm=True, transpose=True)
        k = [1,3,5]
        ids = np.array(ids)   

        # Create empty list to store results 
        result_dict_cos = {}
        result_dict_euc = {}

        for i in k: 
            result_dict_cos[i] = []
            result_dict_euc[i] = []

        for i in os.listdir(self.input):
            img_label = int(self.golden[i])
            result_index_list_cos,_, result_index_list_euc,_ = self.predict_id(self.input+'/'+i, embeddings, k=k)
            for index, j in enumerate(k):
                result_cos = ids[result_index_list_cos[index]]
                result_euc = ids[result_index_list_euc[index]]
                # Append cos result
                if img_label in result_cos:
                    result_dict_cos[j].append(True)
                else:
                    result_dict_cos[j].append(False)
                # Append euc result
                if img_label in result_euc:
                    result_dict_euc[j].append(True)
                else:
                    result_dict_euc[j].append(False)
        for a in k:
            correct_cos = sum(1 if x else 0 for x in result_dict_cos[a]) 
            acc_cos = round(correct_cos*100/len(result_dict_cos[a]),3)
            correct_euc = sum(1 if x else 0 for x in result_dict_euc[a]) 
            acc_euc = round(correct_euc*100/len(result_dict_euc[a]),3)
            print("k=",a, ":", correct_cos, "out of " ,len(result_dict_cos[a]), "\tAccuracy:", acc_cos)
            print("k=",a, ":", correct_euc, "out of " ,len(result_dict_euc[a]), "\tAccuracy:", acc_euc)
            if self.use_clearml:
                print("Uploading")
                self.logger.report_scalar(
                    "accuracy", 'cos', iteration=a, value=acc_cos
                )
                self.logger.report_scalar(
                    "accuracy", 'euc', iteration=a, value=acc_euc
                )
    

    @staticmethod
    def add_eval_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exp_name",
            default='experiment',
            help="Experiment Name"
        )
        parser.add_argument(
            "-i",
            "--input",
            default='test/',
            help="Evaluation Set image folder path"
        )
        parser.add_argument(
            "-e",
            "--emb",
            default='emb/',
            help="Embedding output folder"
        )
        parser.add_argument(
            "-l",
            "--label",
            default='label.json',
            help="Golden Label File Path"
        )
        parser.add_argument(
            "-m",
            "--model_path",
            default='model.pt',
            help="Path & Model Name"
        )
        parser.add_argument(
            "-c",
            "--use_clearml",
            action="store_true",
            help="Connect to ClearML"
        )
        parser.add_argument(
            "-s",
            "--s3",
            action="store_false",
            help="Call to use s3"
        )
        parser.add_argument(
            "--s3_dataset_name",
            default='vggface_exp10',
            help="ClearML Dataset Name"
        )

        return parser

if __name__ == '__main__':
    parser = Evaluate.add_eval_args()
    args = parser.parse_args()
    gen = Evaluate(args)
    gen.evaluate()