
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
# from model.utils import training2 as training
from data.preprocessor import PreProcessor

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import argparse
from clearml import Task, Dataset
from pathlib import Path
import random

random.seed(42)

from data.TripletLossDataset import TripletFaceDataset
from data.triplet_loss import TripletLoss
from data.generate_csv_files import generate_csv_file
from LFW_utils.validate_on_LFW import evaluate_lfw
from LFW_utils.LFWDataset import LFWDataset
from LFW_utils.plot import plot_roc_lfw, plot_accuracy_lfw
from torch.nn.modules.distance import PairwiseDistance
from tqdm import tqdm

PROJECT_NAME = 'facenet'
OUTPUT_URL = 's3://experiment-logging/'


class Experiment(object):

   # should init as arguments here
    def __init__(self, args):
        self.clearml = args.clearml
        if self.clearml:
            # self.clearml_task = Task.get_task(project_name=PROJECT_NAME, task_name='pl_train_triplet2')
            self.clearml_task = Task.init(project_name=PROJECT_NAME, task_name='pl_train_triplet2_'+args.exp_name) # DEBUG
            self.clearml_task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04", 
                docker_setup_bash_script=['pip3 install sklearn', 'pip3 install matplotlib']
            )
            self.clearml_task.execute_remotely(queue_name="compute")
        # print("Init successful")
        self.s3 = args.s3
        self.args = args
        self.data_dir = os.path.join(args.data_dir, '')       # data_dir = 'exp10/train'
        self.batch_size = args.batch_size   # batch_size = 256
        self.epochs = args.epochs           # epochs = 20
 
        self.workers = 0 if os.name == 'nt' else 2
        self.learn_rate = args.learn_rate   
        self.frozen = args.freeze_layers    # 14
        self.margin = args.margin           # 0.2
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_size = args.im_size      # 140
        self.model_path = args.model_path   # path to export model to exp10/model.pt
        self.output_triplets_path = os.path.join(args.output_triplets_path, '')
        Path(self.output_triplets_path).mkdir(parents=True, exist_ok=True)

        self.lfw_batch_size = args.lfw_batch_size
        self.iterations_per_epoch = args.iterations_per_epoch       # 5000
        self.num_human_id_per_batch=args.num_human_id_per_batch     # 32

        if self.s3:
            # Train Dataset
            dataset_name = args.s3_dataset_name
            dataset_project = "datasets/facenet"
            dataset_path = Dataset.get(
                dataset_name=dataset_name, 
                dataset_project=dataset_project
            ).get_local_copy()
            dataset_path = os.path.join(dataset_path, '')
            self.data_dir=os.path.join(dataset_path+self.data_dir, '')
            pretrained_path = Dataset.get(
                dataset_name='resnet_pretrained', 
                dataset_project=dataset_project
            ).get_local_copy()
            self.pretrained_path = os.path.join(pretrained_path, '')+'20180402-114759-vggface2.pt'

            # LFW Dataset
            lfw_name = args.s3_lfw_name
            lfw_path = Dataset.get(
                dataset_name=lfw_name, 
                dataset_project=dataset_project
            ).get_local_copy()
            lfw_path = os.path.join(lfw_path, '')
            self.lfw_pairs = lfw_path+args.lfw_pairs
            self.lfw_dataroot = lfw_path+os.path.join(args.lfw_dataroot, '')
            self.output_path = Dataset.create(dataset_name=args.exp_name+'_models', dataset_project = 'datasets/facenet')
            
            
        else:

            # LFW Dataset
            self.lfw_dataroot = os.path.join(args.lfw_dataroot, '')
            self.lfw_pairs = args.lfw_pairs
            # self.log_path = os.path.join(args.log_path, '')
            # Path(self.log_path).mkdir(parents=True, exist_ok=True)
            # self.plot_path = os.path.join(args.plot_path, '')
            # Path(self.plot_path).mkdir(parents=True, exist_ok=True)
            # Exports


        print("Done Init")

    def get_lfw_dataloader(self):
        lfw_transforms = transforms.Compose([
            transforms.Resize(size=self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])

        lfw_dataloader = DataLoader(
            dataset=LFWDataset(
                dir=self.lfw_dataroot,
                pairs_path=self.lfw_pairs,
                transform=lfw_transforms
            ),
            batch_size=self.lfw_batch_size,
            num_workers=self.workers,
            shuffle=False
        )
        return lfw_dataloader

    def validate_lfw(self, model, lfw_dataloader, epoch):
        model.eval()
        with torch.no_grad():
            l2_distance = PairwiseDistance(p=2)
            distances, labels = [], []

            print("Validating on LFW! ...")
            progress_bar = enumerate(tqdm(lfw_dataloader))

            for batch_index, (data_a, data_b, label) in progress_bar:
                data_a = data_a.cuda()
                data_b = data_b.cuda()

                output_a, output_b = model(data_a), model(data_b)
                distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

                distances.append(distance.cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())

            labels = np.array([sublabel for label in labels for sublabel in label])
            distances = np.array([subdist for distance in distances for subdist in distance])

            true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
            tar, far = evaluate_lfw(
                distances=distances,
                labels=labels,
                far_target=1e-3
            )
            # Print statistics and add to log
            print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
                "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
                "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                        np.mean(accuracy),
                        np.std(accuracy),
                        np.mean(precision),
                        np.std(precision),
                        np.mean(recall),
                        np.std(recall),
                        roc_auc,
                        np.mean(best_distances),
                        np.std(best_distances),
                        np.mean(tar),
                        np.std(tar),
                        np.mean(far)
                    )
            )
            # with open(self.log_path+'lfw_log_triplet.txt', 'a') as f:
            #     val_list = [
            #         epoch,
            #         np.mean(accuracy),
            #         np.std(accuracy),
            #         np.mean(precision),
            #         np.std(precision),
            #         np.mean(recall),
            #         np.std(recall),
            #         roc_auc,
            #         np.mean(best_distances),
            #         np.std(best_distances),
            #         np.mean(tar)
            #     ]
            #     log = '\t'.join(str(value) for value in val_list)
            #     f.writelines(log + '\n')

        # try:
        #     # Plot ROC curve
        #     plot_roc_lfw(
        #         false_positive_rate=false_positive_rate,
        #         true_positive_rate=true_positive_rate,
        #         figure_name=self.plot_path+"roc_plots/roc_epoch_{}_triplet.png".format(epoch)
        #     )
        #     # Plot LFW accuracies plot
        #     plot_accuracy_lfw(
        #         log_file=self.log_path+"lfw_log_triplet.txt",
        #         epochs=epoch,
        #         figure_name=self.plot_path+"accuracies_plots/lfw_accuracies_epoch_{}_triplet.png".format(epoch)
        #     )
        # except Exception as e:
        #     print(e)

        return best_distances, accuracy, precision, recall, roc_auc


    def forward_pass(self, imgs, model, batch_size):
        imgs = imgs.to(self.device)
        embeddings = model(imgs)

        # Split the embeddings into Anchor, Positive, and Negative embeddings
        anc_embeddings = embeddings[:batch_size]
        pos_embeddings = embeddings[batch_size: batch_size * 2]
        neg_embeddings = embeddings[batch_size * 2:]

        return anc_embeddings, pos_embeddings, neg_embeddings, model

    def run_experiment(self):

        if self.clearml:
            logger = self.clearml_task.get_logger()
        
        print('Running on device: {}'.format(self.device))
        print("data dir is", self.data_dir)
        mtcnn = MTCNN(
            image_size=224, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        # Use MTCNN to preprocess & crop images
        preprocess = PreProcessor(mtcnn, self.data_dir[:-1], self.batch_size)
        dataset = preprocess.crop_img()
        # print(self.data_dir[:-1]+ '_cropped')
        df = generate_csv_file(self.data_dir[:-1]+ '_cropped')
        # df.to_csv(path_or_buf=csv_name, index=False)
        # Init Resnet model
        if self.s3:
            resnet = InceptionResnetV1(
                classify=False,
                pretrained='vggface2',
                s3_path=self.pretrained_path
            ).to(self.device)
        else:
            resnet = InceptionResnetV1(
                classify=False,
                pretrained='vggface2'
            ).to(self.device)

        # Freeze most layers
        count=0
        for child in resnet.children():
            if count<=self.frozen:
                for param in child.parameters():
                    param.requires_grad = False
            count+=1
        
        optimizer = optim.Adam(resnet.parameters(), lr=self.learn_rate)
        scheduler = MultiStepLR(optimizer, [2, 3])

        trans = transforms.Compose([
            transforms.Resize(size=self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.6071, 0.4609, 0.3944],
                std=[0.2457, 0.2175, 0.2129]
            )
        ])

        lfw_dataloader = self.get_lfw_dataloader()

        print('\n\nInitial')
        print('-' * 10)
        best_distances, accuracy, precision, recall, roc_auc = self.validate_lfw(
            model=resnet,
            lfw_dataloader=lfw_dataloader,
            epoch=0
        )
        if self.clearml:
            logger.report_scalar("LFW acc", "A", iteration=-1, value=np.mean(accuracy))
            logger.report_scalar("LFW precision", "A", iteration=-1, value=np.mean(precision))
            logger.report_scalar("LFW recall", "A", iteration=-1, value=np.mean(recall))
            logger.report_scalar("LFW ROC", "A", iteration=-1, value=roc_auc)                  

        min_loss = 99
        for epoch in range(self.epochs):
            num_valid_training_triplets = 0
            l2_distance = PairwiseDistance(p=2)
            _training_triplets_path = None
            epoch_loss = torch.zeros(1)
            print('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)
            # Train Dataloader is reloaded every epoch
            train_dataloader = DataLoader(
                dataset=TripletFaceDataset(
                    root_dir=self.data_dir[:-1]+'_cropped',
                    training_dataset_df=df,
                    num_triplets=self.iterations_per_epoch * self.batch_size,
                    num_human_identities_per_batch=self.num_human_id_per_batch,
                    triplet_batch_size=self.batch_size,
                    epoch=epoch,
                    training_triplets_path=_training_triplets_path,
                    output_triplets_path = self.output_triplets_path,
                    transform=trans
                ),
                batch_size=self.batch_size,
                num_workers=self.workers,
                shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
            )
            
            resnet.train()

            for batch_idx, (batch_sample) in enumerate(tqdm(train_dataloader)):
                batch_loss = torch.zeros(1)
                anc_imgs = batch_sample['anc_img']
                pos_imgs = batch_sample['pos_img']
                neg_imgs = batch_sample['neg_img']
                all_imgs = torch.cat((anc_imgs, pos_imgs, neg_imgs))
                
                anc_embeddings, pos_embeddings, neg_embeddings, resnet = self.forward_pass(
                    imgs=all_imgs,
                    model=resnet,
                    batch_size=self.batch_size
                )
                
                pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
                neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)
                # Hard Negative
                all = (neg_dists - pos_dists < self.margin).cpu().numpy().flatten()
                valid_triplets = np.where(all == 1)
                
                anc_valid_embeddings = anc_embeddings[valid_triplets]
                pos_valid_embeddings = pos_embeddings[valid_triplets]
                neg_valid_embeddings = neg_embeddings[valid_triplets]
                
                batch_loss = TripletLoss(margin=self.margin).forward(
                    anchor=anc_valid_embeddings,
                    positive=pos_valid_embeddings,
                    negative=neg_valid_embeddings
                )
                
                num_valid_training_triplets += len(anc_valid_embeddings)
                optimizer.zero_grad()
                batch_loss.backward()
                epoch_loss = epoch_loss+batch_loss.to('cpu')
                optimizer.step()
            scheduler.step()
            print('Epoch {}:\tNumber of valid training triplets in epoch: {}'.format(epoch,num_valid_training_triplets))
            epoch_loss = epoch_loss/len(train_dataloader)

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(resnet.state_dict(), self.model_path[:-3]+'_epoch_{}.pt'.format(epoch))
                if self.s3:
                    self.output_path.add_files(self.model_path[:-3]+'_epoch_{}.pt'.format(epoch))
            if self.clearml:
                logger.report_scalar("loss (by epoch)", "train", iteration=epoch, value=epoch_loss.item())                
            # with open(self.log_path+'log_triplet.txt', 'a') as f:
            #     val_list = [
            #         epoch,
            #         num_valid_training_triplets
            #     ]
            #     log = '\t'.join(str(value) for value in val_list)
            #     f.writelines(log + '\n')    
                
            best_distances, accuracy, precision, recall, roc_auc = self.validate_lfw(
                model=resnet,
                lfw_dataloader=lfw_dataloader,
                epoch=epoch
            )
            if self.clearml:
                logger.report_scalar("LFW acc", "A", iteration=epoch, value=np.mean(accuracy))
                logger.report_scalar("LFW precision", "A", iteration=epoch, value=np.mean(precision))
                logger.report_scalar("LFW recall", "A", iteration=epoch, value=np.mean(recall))
                logger.report_scalar("LFW ROC", "A", iteration=epoch, value=roc_auc)      

        torch.save(resnet.state_dict(), self.model_path)
        if self.s3:
            self.output_path.add_files(self.model_path)
            self.output_path.upload(output_url='s3://experiment-logging/')
            self.output_path.finalize()
            self.output_path.publish()


    @staticmethod
    def add_experiment_args():

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--exp_name",
            default='experiment',
            help="Experiment Name"
        )
        # Train Dataset Args
        parser.add_argument(
            "-d",
            "--data_dir",
            default='data/train',
            help="Training Dataset Folder Path"
        )
        #  Model Param Args
        parser.add_argument(
            "-b",
            "--batch_size",
            default=256,
            type=int,
            help="Data Loader Batch Size"
        )
        parser.add_argument(
            "-e",
            "--epochs",
            default=20,
            type=int,
            help="Number of Epochs to train for"
        )
        parser.add_argument(
            "-f",
            "--freeze_layers",
            default=14,
            type=int,
            help="Number of layers to freeze (Max 17, i.e. all layers frozen)"
        )
        parser.add_argument(
            "-p",
            "--model_path",
            default='model.pt',
            help="Path & Model Name"
        )
        parser.add_argument(
            "-r",
            "--learn_rate",
            default=0.001,
            type=float
        )
        parser.add_argument(
            "-m",
            "--margin",
            help='Triplet Loss Margin',
            default=0.2,
            type=float
        )
        parser.add_argument(
            "--im_size",
            help='Image Size',
            default=140,
            type=int
        )
        parser.add_argument(
            "--iterations_per_epoch",
            help='No. of Iterations Per Epoch',
            default=5000,
            type=int
        )
        parser.add_argument(
            "--num_human_id_per_batch",
            help='Number of Identities per batch',
            default=32,
            type=int
        )
        parser.add_argument(
            "--output_triplets_path",
            default='generated_triplets/',
            help="Path to output the generated triplets"
        )
        # Eval (LFW) args
        parser.add_argument(
            "--lfw_dataroot",
            default='LFW_utils/lfw_224/',
            help="Path to cropped lfw dataset"
        )
        parser.add_argument(
            "--lfw_pairs",
            default='LFW_utils/LFW_pairs.txt',
            help="Path to lfw pair labels"
        )
        parser.add_argument(
            "--lfw_batch_size",
            help='LFW batch size',
            default=200,
            type=int
        )
        parser.add_argument(
            "--log_path",
            default='LFW_utils/logs/',
            help="Path to log folder"
        )        
        parser.add_argument(
            "--plot_path",
            default='LFW_utils/plots/',
            help="Path to plot folder"
        )       

        # ClearML
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
        parser.add_argument(
            "--s3_lfw_name",
            default='lfw_eval',
            help="LFW Eval Dataset Name"
        )              


        return parser

if __name__ == '__main__':
    parser = Experiment.add_experiment_args()
    args = parser.parse_args()
    exp = Experiment(args)
    exp.run_experiment()