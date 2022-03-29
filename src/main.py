import argparse
import os
import experiment
from clearml import Task, Logger
# from clearml import Task
# from aiplatform.config import cfg as aip_cfg

if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser = experiment.Experiment.add_experiment_args()


    args = parser.parse_args()
    # task = Task.init(project_name='facenet', task_name=args.task_name)
    logger = task.get_logger()
    # task = None

    exp = experiment.Experiment(args)
    exp.run_experiment()


    # exp.create_torchscript_model('class_model_v2.ckpt')
    # exp.create_torchscript_cpu_model('id_model4.ckpt')
