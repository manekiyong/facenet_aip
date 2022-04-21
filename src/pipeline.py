from clearml import Task
from clearml.automation import PipelineController

PROJECT_NAME = 'facenet'
PIPELINE_NAME = 'tamer_256_20_1e3_0.2_15_5000_64'

params = {
    'triplet':True,                                             # stage 1t
    'data_dir':'data/exp6/train',                              # stage 1, 2
    'id_csv': 'data/exp6/exp6.csv',                           # stage 1t
    'batch_size':256,                                           # stage 1
    'epochs':20,                                                # stage 1
    'learn_rate':1e-3,                                          # stage 1t
    'margin':0.2,                                               # stage 1t
    'freeze_layers':15,                                         # stage 1
    'iterations_per_epoch': 5000,                               # stage 1t
    'num_human_id_per_batch': 64,                               # stage 1t
    'output_triplets_path': 'data/exp6/generated_triplets/',   # stage 1t
    'model_path':'data/exp6/tamer_256_20_1e3_0.2_15_5000_64.pt',  # stage 1, 2, 3
    'emb_dir':'data/exp6/emb',                                 # stage 2, 3
    'eval_dir':'data/exp6/test',                               # stage 3
    'label_path':'data/exp6/label.json',                       #stage 3
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
    if params['triplet']:
        #Add triplet step
        pipe.add_step(name='train_model',
            base_task_project=PROJECT_NAME,
            base_task_name='pl_train_triplet2',
            parameter_override={
                'Args/clearml': True,
                'Args/data_dir': params['data_dir'],
                'Args/id_csv': params['id_csv'],
                'Args/batch_size': params['batch_size'],
                'Args/epochs': params['epochs'],
                'Args/freeze_layers': params['freeze_layers'],
                'Args/model_path': params['model_path'],
                'Args/margin': params['margin'],
                'Args/learn_rate': params['learn_rate'],
                'Args/iterations_per_epoch': params['iterations_per_epoch'],
                'Args/num_human_id_per_batch': params['num_human_id_per_batch'],
                'Args/output_triplets_path': params['output_triplets_path']
            }
        )
    else:
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
    pipe.add_step(name='evaluate',
        parents=['generate_embedding', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_evaluate',
        parameter_override={
            'Args/use_clearml': True,
            'Args/input': params['eval_dir'],
            'Args/emb': params['emb_dir'],
            'Args/label': params['label_path'],
            'Args/model_path': params['model_path'],
        }
    )

    
    pipe.set_default_execution_queue('default')

    # pipe.add_step(name='generate_embedding', parents=['train_model', ], base_task_project=PROJECT_NAME, base_task_name=args.task_name+' generate')

    # for debugging purposes use local jobs
    pipe.start_locally(True)

    # Starting the pipeline (in the background)
    # pipe.start()

    print('done')
