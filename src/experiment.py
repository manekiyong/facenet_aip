
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
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
import argparse

# TODO: check tensor types


def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()




class Experiment(object):

   # should init as arguments here
    def __init__(self, args, clearml_task=None):

        self.clearml_task = clearml_task
        self.args = args
        self.data_dir = args.data_dir       # data_dir = 'exp4/train'
        self.batch_size = args.batch_size   # batch_size = 32
        self.epochs = args.epochs           # epochs = 20
        self.model_path = args.model_path   # path to export model to
        self.workers = 0 if os.name == 'nt' else 8


    def run_experiment(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        print('Running on device: {}'.format(device))
        mtcnn = MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=device
        )
        # Use MTCNN to preprocess & crop images
        preprocess = PreProcessor(mtcnn, self.args)
        dataset = preprocess.crop_img(training.collate_pil)

        # Init Resnet model
        resnet = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=len(dataset.class_to_idx)
        ).to(device)

        # Freeze most layers
        count=0
        for child in resnet.children():
            if count<=15:
                for param in child.parameters():
                    param.requires_grad = False
            count+=1
        
        optimizer = optim.Adam(resnet.parameters(), lr=0.001)
        scheduler = MultiStepLR(optimizer, [5, 10])

        trans = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization
        ])

        dataset = datasets.ImageFolder(self.data_dir + '_cropped', transform=trans)
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
        loss_fn = torch.nn.CrossEntropyLoss()
        metrics = {
            'fps': training.BatchTimer(),
            'acc': training.accuracy
        }

        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        print('\n\nInitial')
        print('-' * 10)
        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        for epoch in range(self.epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, self.epochs))
            print('-' * 10)

            resnet.train()
            training.pass_epoch(
                resnet, loss_fn, train_loader, optimizer, scheduler,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

            resnet.eval()
            training.pass_epoch(
                resnet, loss_fn, val_loader,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer
            )

        writer.close()
        torch.save(resnet.state_dict(), self.model_path)

    @staticmethod
    def add_experiment_args(parent_parser):

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
            "-m",
            "--model_path",
            default='model.pt',
            help="Path & Model Name"
        )

        return parser

    # @staticmethod
    # def create_torchscript_model(model_name):
    #     model = Seq2Seq.load_from_checkpoint(os.path.join(
    #         cfg['train']['checkpoint_dir'], model_name))

    #     model.eval()

    #     # remove_empty_attributes(model)
    #     # print(vars(model._modules['input_mapper']))
    #     # print('These attributes should have been removed', remove_attributes)
    #     script = model.to_torchscript()
    #     torch.jit.save(script, os.path.join(
    #         cfg['train']['checkpoint_dir'], "model.pt"))

#     @staticmethod
#     def create_torchscript_cpu_model(model_name):
#         model = Seq2Seq.load_from_checkpoint(os.path.join(
#             cfg['train']['checkpoint_dir'], model_name))

#         model.to('cpu')
#         model.eval()

#         # remove_empty_attributes(model)
#         # print(vars(model._modules['input_mapper']))
#         # print('These attributes should have been removed', remove_attributes)
#         script = model.to_torchscript()
#         torch.jit.save(script, os.path.join(
#             cfg['train']['checkpoint_dir'], "model_cpu.pt"))


# def remove_empty_attributes(module):
#     remove_attributes = []
#     for key, value in vars(module).items():
#         if value is None:

#             if key == 'trainer' or '_' == key[0]:
#                 remove_attributes.append(key)
#         elif key == '_modules':
#             for mod in value.keys():

#                 remove_empty_attributes(value[mod])
#     print('To be removed', remove_attributes)
#     for key in remove_attributes:

#         delattr(module, key)


# class CustomCheckpoint(ModelCheckpoint):
#     CHECKPOINT_JOIN_CHAR = "-"
#     CHECKPOINT_NAME_LAST = "last"
#     FILE_EXTENSION = ".ckpt"
#     STARTING_VERSION = 1


#     def __init__(
#         self,
#         task_name = None,
#         dirpath: Optional[Union[str, Path]] = None,
#         filename: Optional[str] = None,
#         monitor: Optional[str] = None,
#         verbose: bool = False,
#         save_last: Optional[bool] = None,
#         save_top_k: Optional[int] = None,
#         save_weights_only: bool = False,
#         mode: str = "min",
#         auto_insert_metric_name: bool = True,
#         every_n_train_steps: Optional[int] = None,
#         every_n_val_epochs: Optional[int] = None,
#         period: Optional[int] = None,
#     ):
#         super().__init__(
#             dirpath,
#             filename,
#             monitor,
#             verbose,
#             save_last,
#             save_top_k,
#             save_weights_only,
#             mode,
#             auto_insert_metric_name,
#             every_n_train_steps,
#             every_n_val_epochs,
#             period,
#         )
#         S3_PATH = os.path.join(aip_cfg.s3.model_artifact_path,task_name)
#         latest_model_name = 'latest_model.ckpt'
#         best_model_name = 'best_model.ckpt'


#     def _save_model(self, trainer: 'pl.Trainer', filepath: str) -> None:
#         try:
#             if trainer.training_type_plugin.rpc_enabled:
#                 # RPCPlugin manages saving all model states
#                 # TODO: the rpc plugin should wrap trainer.save_checkpoint
#                 # instead of us having to do it here manually
#                 trainer.training_type_plugin.rpc_save_model(trainer, self._do_save, filepath)
#             else:
#                 self._do_save(trainer, filepath)
#         except:
#             self._do_save(trainer, filepath)

#         # call s3 function here to upload file to s3 using filepath
#         # self.clearml_task.upload_file(filepath,'https://ecs.dsta.ai/bert_finetune_lm/artifact/saved_model.ckpt')

#         # folder = 'uncased' if self.use_uncased else 'cased'

#         print('uploading model checkpoint to S3...')
#         s3_utils = S3Utils(aip_cfg.s3.bucket,aip_cfg.s3.model_artifact_path)


#         s3_utils.s3_upload_file(filepath,os.path.join(S3_PATH,self.latest_model_name))

#         best_model_path = self.best_model_path
#         print("\nBEST MODEL PATH: ", best_model_path, "\n")


#         print('uploading best model checkpoint to S3...')

#         s3_utils.s3_upload_file(best_model_path,os.path.join(S3_PATH,self.best_model_name))
