from model.inception_resnet_v1 import InceptionResnetV1
from model.mtcnn import MTCNN

import torch
import numpy as np
import os
import json
import argparse
from pathlib import Path
from PIL import Image
from clearml import Task


PROJECT_NAME = 'facenet'

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()


class Evaluate(object):

   # should init as arguments here
    def __init__(self, args):
        self.clearml = args.clearml
        if self.clearml:
            self.clearml_task = Task.get_task(project_name=PROJECT_NAME, task_name='pl_evaluate')
        self.input = os.path.join(args.input, '')
        self.emb = os.path.join(args.emb, '')
        self.resnet = InceptionResnetV1(pretrained=None, classify=False)
        self.resnet.load_state_dict(torch.load(args.model_path), strict=False)
        self.resnet.eval()
        self.mtcnn = MTCNN(image_size=160, margin=0, device='cuda', keep_all=True)
        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        with open(args.label) as json_file:
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
        temp_sim = []
        img = Image.open(img_path)
        img_cropped, prob = self.mtcnn(img, return_prob=True)
        if prob[0] != None:
            max_val = np.argmax(prob)
            img_embedding = self.resnet(img_cropped)
            img_embedding = img_embedding[max_val]
        else:
            return [[-1]]*len(k), [[-1]]*len(k)
        temp_sim = torch.matmul(img_embedding, emb_list)
        indices_list = []
        values_list = []
        for i in k:
            topk = torch.topk(temp_sim, i)
            indices_list.append(topk.indices.tolist())
            values_list.append(topk.values.tolist())
        return indices_list, values_list


    def evaluate(self):
        embeddings, ids = self.load_emb(self.emb, norm=True, transpose=True)
        k = [1,2,3]
        if self.clearml:
            logger = self.clearml_task.get_logger()
        ids = np.array(ids)   

        ############
        logger.report_text("meow ")   
        for i in range(100):
            logger.report_scalar(
                "unified graph", "series A", iteration=i, value=1./(i+1)
            )
        ############


        # Create empty list to store results 
        result_dict = {}
        for i in k: 
            result_dict[i] = []

        for i in os.listdir(self.input):
            img_label = int(self.golden[i])
            # print(i, img_label)
            result_index_list, conf_list = self.predict_id(self.input+'/'+i, embeddings, k=k)
            for index, j in enumerate(k):
                result = ids[result_index_list[index]]
                # print("Ground Truth:", img_label, "\tPrediction:", result, "\tConfidence:", conf_list[index])
                if img_label in result:
                    result_dict[j].append(True)
                else:
                    result_dict[j].append(False)

        for i in k:
            correct = sum(1 if x else 0 for x in result_dict[i]) 
            acc = round(correct*100/len(result_dict[i]),3)
            print("k=",i, ":", correct, "out of " ,len(result_dict[i]), "\tAccuracy:", acc)
            if self.clearml:
                print("Uploading")
                logger.report_text("meow ")
                logger.report_scalar("accuracy", 'temp', iteration=i, value=acc)  


    @staticmethod
    def add_eval_args():
        parser = argparse.ArgumentParser()
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
            "--clearml",
            action="store_true",
            help="Connect to ClearML"
        )

        return parser

if __name__ == '__main__':
    parser = Evaluate.add_eval_args()
    args = parser.parse_args()
    gen = Evaluate(args)
    gen.evaluate()