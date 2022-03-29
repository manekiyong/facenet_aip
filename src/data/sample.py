import argparse
import json
import os
import random
import shutil
import pandas as pd
from pathlib import Path
random.seed(42)


class Sample():

    def __init__(self, args):
        self.metadata = args.metadata
        self.num = args.num
        self.min = args.min
        self.max = args.max
        self.reserved = args.reserved
        self.test = args.test
        self.output = args.output
        self.annotation = args.annotation

    def sample(self):
        df=pd.read_csv(self.metadata)
        max_no_of_id = len(df['id'].unique())
        ids = []
        label_dict={}

        Path(self.output).mkdir(parents=True, exist_ok=True)
        Path(self.test).mkdir(parents=True, exist_ok=True)
        while(len(os.listdir(self.output))<self.num):
            rand_list = []
            for i in range(self.num):
                rand_id = random.randint(0, max_no_of_id-1)
                while rand_id in ids: # Resample if id is already processed/found
                    rand_id = random.randint(0, max_no_of_id-1)
                rand_list.append(rand_id)

            pop_list = []
            for i, j in enumerate(rand_list): 
                temp_df = df[df['id']==j]
                if len(temp_df) < self.min or len(temp_df) > self.max:
                    pop_list.append(i)
            for i in reversed(pop_list):
                rand_list.pop(i)

            # Pop excess ids
            while (len(os.listdir(self.output))+len(rand_list)>self.num):
                rand_list.pop(random.randint(0, len(rand_list)-1))

            for i in rand_list:
                ids.append(i)
                temp_df = df[df['id']==i]
                Path(self.output+str(i)).mkdir(parents=True, exist_ok=True)
                for j, rows in temp_df.iterrows():
                    shutil.copy2(rows['image_id'], self.output+str(i)+'/')
            print(len(os.listdir(self.output)), "ids done...")    
        
        # Holdout set
        print("Populating Holdout set...")
        for ids in os.listdir(self.output):
            folder_path = self.output+ids+'/'
            folder_content = os.listdir(folder_path)
            #if folder has less images than holdout, terminate
            if self.reserved >= len(folder_content):
                print("Not enough images in folder!")
                return 
            eval_content = random.sample(folder_content, self.reserved)
            for i in eval_content:
                shutil.copy2(folder_path+str(i), self.test+str(i))
                os.remove(folder_path+str(i))
                label_dict[i]=ids
        with open(self.annotation, 'w') as fp:
            json.dump(label_dict, fp)


    @staticmethod
    def add_args():
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "-m",
                "--metadata",
                default='full.csv',
                help="Dataset metadata"
            )
        parser.add_argument(
            "-n",
            "--num",
            default=200,
            type=int,
            help="Number of ID to sample"
        )
        parser.add_argument(
            "-l",
            "--min",
            default=20,
            type=int,
            help="Min. no. of image of each ID"
        )
        parser.add_argument(
            "-u",
            "--max",
            default=30,
            type=int,
            help="Max. no. of image of each ID"
        )
        parser.add_argument(
            "-r",
            "--reserved",
            default=0,
            type=int,
            help="Hold out images for evaluation"
        )
        parser.add_argument(
            "-t",
            "--test",
            default='test/',
            help="Test set image folder path"
        )
        parser.add_argument(
            "-o",
            "--output",
            default='train/',
            help="Output Path"
        )
        parser.add_argument(
            "-a",
            "--annotation",
            default='label.json',
            help="Test Set Annotation Label"
        )
        return parser
