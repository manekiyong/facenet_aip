from clearml import Task
from clearml.automation import PipelineController
import argparse
import experiment

PROJECT_NAME = 'facenet'
PIPELINE_NAME = 'Experiment_0'

params = {
    'data_dir':'data/exp1/train',   # stage 1, 2
    'batch_size':32,                # stage 1
    'epochs':20,                    # stage 1
    'freeze_layers':15,             # stage 1
    'model_path':'pl.pt',           # stage 1, 2, 3
    'emb_dir':'data/exp1/emb',      # stage 2, 3
    'eval_dir':'data/exp1/test',    # stage 3
    'label_path':'data/exp1/label.json',    #stage 3
    'topk':1                        # stage 3
}



if __name__ == '__main__':
    # Connecting ClearML with the current pipeline,
    # from here on everything is logged automatically



    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PROJECT_NAME,
        version='0.0.1',
        add_pipeline_tags=False,
    )
    print("Pipe created")
    pipe.add_step(name='train_model',
        base_task_project=PROJECT_NAME,
        base_task_name='pl_train',
        parameter_override={
            'Args/clearml': True,
            'Args/batch_size': params['batch_size'],
            'Args/data_dir': params['data_dir'],
            'Args/epochs': params['epochs'],
            'Args/freeze_layers': params['freeze_layers'],
            'Args/model_path': params['model_path']
            }
    )
    pipe.add_step(name='generate_embedding',
        parents=['train_model', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_generate',
        parameter_override={
            'Args/input': params['data_dir'],
            'Args/output': params['emb_dir'],
            'Args/model_path': params['model_path']
            }
    )
    pipe.add_step(name='evaluate_1',
        parents=['generate_embedding', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_evaluate',
        parameter_override={
            'Args/clearml': True,
            'Args/input': params['eval_dir'],
            'Args/emb': params['emb_dir'],
            'Args/label': params['label_path'],
            'Args/model_path': params['model_path'],
            'Args/topk': 1
            }
    )
    pipe.add_step(name='evaluate_3',
        parents=['generate_embedding', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_evaluate',
        parameter_override={
            'Args/clearml': True,
            'Args/input': params['eval_dir'],
            'Args/emb': params['emb_dir'],
            'Args/label': params['label_path'],
            'Args/model_path': params['model_path'],
            'Args/topk': 3
            }
    )
    pipe.add_step(name='evaluate_5',
        parents=['generate_embedding', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_evaluate',
        parameter_override={
            'Args/clearml': True,
            'Args/input': params['eval_dir'],
            'Args/emb': params['emb_dir'],
            'Args/label': params['label_path'],
            'Args/model_path': params['model_path'],
            'Args/topk': 5
            }
    )
    
    pipe.set_default_execution_queue('default')

    # pipe.add_step(name='generate_embedding', parents=['train_model', ], base_task_project=PROJECT_NAME, base_task_name=args.task_name+' generate')

    # for debugging purposes use local jobs
    pipe.start_locally(True)

    # Starting the pipeline (in the background)
    # pipe.start()

    print('done')
