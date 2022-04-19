
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
from model.utils import training2 as training
from data.preprocessor import PreProcessor

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
from clearml import Task
import random

random.seed(42)

PROJECT_NAME = 'facenet'

class MyDataset(datasets.ImageFolder):
    def __init__(self, path, transform):
        super().__init__(path, transform=transform)
        self.transform = transform

    def __getitem__(self, index):
        dataset_len = super().__len__()
        im, label = super().__getitem__(index)
        pos_im, pos_label = super().__getitem__(random.randint(0, dataset_len-1))
        while torch.equal(im, pos_im) or pos_label != label:
            pos_im, pos_label = super().__getitem__(random.randint(0, dataset_len-1))
        neg_im, neg_label = super().__getitem__(random.randint(0, dataset_len-1))
        while neg_label == label:
            neg_im, neg_label = super().__getitem__(random.randint(0, dataset_len-1))
        anchor = torch.permute(self.transform(im), (1,2,0))
        pos = torch.permute(self.transform(pos_im), (1,2,0))
        neg = torch.permute(self.transform(neg_im), (1,2,0))
        return anchor, label, pos, neg


class Experiment(object):

   # should init as arguments here
    def __init__(self, args):
        self.clearml = args.clearml
        if self.clearml:
            self.clearml_task = Task.get_task(project_name=PROJECT_NAME, task_name='pl_train_triplet')
        # print("Init successful")
        self.args = args
        self.data_dir = os.path.join(args.data_dir, '')       # data_dir = 'exp9/train'
        self.batch_size = args.batch_size   # batch_size = 32
        self.epochs = args.epochs           # epochs = 20
        self.model_path = args.model_path   # path to export model to exp9/model.pt
        self.workers = 0 if os.name == 'nt' else 8
        self.learn_rate = args.learn_rate
        self.frozen = args.freeze_layers
        self.margin = args.margin
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Done Init")


    def run_experiment(self):
        # self.clearml_task.execute_remotely()
        # print("Execute remote successful")
        
        if self.clearml:
            logger = self.clearml_task.get_logger()
        
        print('Running on device: {}'.format(self.device))
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )
        # Use MTCNN to preprocess & crop images
        preprocess = PreProcessor(mtcnn, self.args)
        dataset = preprocess.crop_img()

        # Init Resnet model
        resnet = InceptionResnetV1(
            classify=False,
            pretrained='vggface2',
            num_classes=len(dataset.class_to_idx)
        ).to(self.device)

        # Freeze most layers
        count=0
        for child in resnet.children():
            if count<=self.frozen:
                for param in child.parameters():
                    param.requires_grad = False
            count+=1
        
        optimizer = optim.Adam(resnet.parameters(), lr=self.learn_rate)
        scheduler = MultiStepLR(optimizer, [5, 10])

        trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])

        dataset = MyDataset(self.data_dir[:-1]+ '_cropped', transform=trans)
        img_inds = np.arange(len(dataset))
        np.random.shuffle(img_inds)
        train_inds = img_inds[:int(0.8 * len(img_inds))]
        val_inds = img_inds[int(0.8 * len(img_inds)):]

        train_loader = DataLoader(
            dataset,
            num_workers=self.workers,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(train_inds)
        )
        val_loader = DataLoader(
            dataset,
            num_workers=self.workers,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(val_inds)
        )
        loss_fn = torch.nn.TripletMarginLoss(margin=self.margin,p=2)
        metrics = {
            'fps': training.BatchTimer(),
            'acc': training.accuracy
        }
        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        print('\n\nInitial')
        print('-' * 10)
        # resnet.eval()
        # training.pass_epoch(
        #     resnet, loss_fn, val_loader,
        #     batch_metrics=metrics, show_running=True, device=self.device,
        #     writer=writer
        # )

        for epoch in range(self.epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)

            resnet.train()
            # Set batch norm layers to eval to prevent var and mean from shifting
            for layer in resnet.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.eval()
            resnet.classify = False
            train_loss, train_metric = training.pass_epoch(
                resnet, loss_fn, train_loader, optimizer, scheduler,
                batch_metrics=metrics, show_running=True, device=self.device,
                writer=writer
            )
            # resnet.classify = True
            # resnet.eval()
            # val_loss, val_metric = training.pass_epoch(
            #     resnet, loss_fn, val_loader,
            #     batch_metrics=metrics, show_running=True, device=self.device,
            #     writer=writer
            # )
            if self.clearml:
                logger.report_scalar("acc (by epoch)", "train", iteration=epoch, value=train_metric['acc'].item())
                # logger.report_scalar("acc (by epoch)", "eval", iteration=epoch, value=val_metric['acc'].item())
                logger.report_scalar("loss (by epoch)", "train", iteration=epoch, value=train_loss.item())
                # logger.report_scalar("loss (by epoch)", "eval", iteration=epoch, value=val_loss.item())                
        writer.close()
        torch.save(resnet.state_dict(), self.model_path)

    @staticmethod
    def add_experiment_args():

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-d",
            "--data_dir",
            default='data/train',
            help="Training Dataset Folder Path"
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            default=32,
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
            "-c",
            "--clearml",
            action="store_false",
            help="Connect to ClearML"
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
            default=1.0,
            type=float
        )
        

        return parser

if __name__ == '__main__':
    parser = Experiment.add_experiment_args()
    args = parser.parse_args()
    exp = Experiment(args)
    exp.run_experiment()