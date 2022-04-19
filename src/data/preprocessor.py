from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def collate_pil(x): 
        out_x, out_y = [], [] 
        for xx, yy in x: 
            out_x.append(xx) 
            out_y.append(yy) 
        return out_x, out_y 

class PreProcessor():

    def __init__(self, mtcnn, args):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.mtcnn = mtcnn
        self.workers = 0 if os.name == 'nt' else 8

    
    def crop_img(self):
        dataset = datasets.ImageFolder(self.data_dir, transform=transforms.Resize((512, 512)))
        dataset.samples = [
            (p, p.replace(self.data_dir, self.data_dir + '_cropped'))
                for p, _ in dataset.samples
        ]
        loader = DataLoader(
            dataset,
            num_workers=self.workers,
            batch_size=self.batch_size,
            collate_fn=collate_pil
        )
        for i, (x, y) in enumerate(loader):
            self.mtcnn(x, save_path=y, return_prob=True)
            print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
        return dataset
