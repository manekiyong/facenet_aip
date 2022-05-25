from clearml import Task
from clearml.automation import PipelineController

PROJECT_NAME = 'facenet'
PIPELINE_NAME = 'tamer_exp11_b256_e1_i10000_id64_f15'

params = {
    'exp_name':PIPELINE_NAME,
    'triplet':True,                                             # stage 1t
    'data_dir':'train/',                              # stage 1, 2
    'batch_size':256,                                           # stage 1
    'epochs':1,                                                # stage 1
    'learn_rate':1e-3,                                          # stage 1t
    'margin':0.2,                                               # stage 1t
    'freeze_layers':15,                                         # stage 1
    'iterations_per_epoch': 10000,                               # stage 1t
    'num_human_id_per_batch': 64,                               # stage 1t
    'semihard': False,
    'output_triplets_path': 'generated_triplets/',   # stage 1t
    'model_path':PIPELINE_NAME+'.pt',  # stage 1, 2, 3
    'emb_dir':'emb/',                                 # stage 2, 3
    'eval_dir':'test/',                               # stage 3
    'label_path':'label.json',                       #stage 3
    'remote': True,
    's3':True,
    's3_dataset_name':'celeba_exp11',
    's3_lfw_name':'lfw_eval',
    'lfw_dataroot':'lfw_224',
    'lfw_pairs':'LFW_pairs.txt'
}



if __name__ == '__main__':
    # Connecting ClearML with the current pipeline,
    # from here on everything is logged automatically

    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=PROJECT_NAME,
        version='0.1',
        add_pipeline_tags=False,
    )
    print("Pipe created")
    if params['triplet']:
        #Add triplet step
        pipe.add_step(name='train_model',
            base_task_project=PROJECT_NAME,
            base_task_name='pl_train_triplet2_'+PIPELINE_NAME,
            parameter_override={
                'Args/clearml': True,
                'Args/remote': params['remote'],
                'Args/data_dir': params['data_dir'],
                'Args/batch_size': params['batch_size'],
                'Args/epochs': params['epochs'],
                'Args/freeze_layers': params['freeze_layers'],
                'Args/model_path': params['model_path'],
                'Args/margin': params['margin'],
                'Args/learn_rate': params['learn_rate'],
                'Args/iterations_per_epoch': params['iterations_per_epoch'],
                'Args/num_human_id_per_batch': params['num_human_id_per_batch'],
                'Args/output_triplets_path': params['output_triplets_path'],
                'Args/semihard': params['semihard'],
                'Args/s3': params['s3'],
                'Args/s3_dataset_name': params['s3_dataset_name'],
                'Args/s3_lfw_name': params['s3_lfw_name'],
                'Args/lfw_dataroot': params['lfw_dataroot'],
                'Args/lfw_pairs': params['lfw_pairs'],
                'Args/exp_name':params['exp_name']

            },
            execution_queue='compute'
        )
    else:
        pipe.add_step(name='train_model',
            base_task_project=PROJECT_NAME,
            base_task_name='pl_train',
            parameter_override={
                'Args/clearml': True,
                'Args/remote': params['remote'],
                'Args/batch_size': params['batch_size'],
                'Args/data_dir': params['data_dir'],
                'Args/epochs': params['epochs'],
                'Args/freeze_layers': params['freeze_layers'],
                'Args/model_path': params['model_path'],
                'Args/s3': params['s3']
            }
        )
    pipe.add_step(name='generate_embedding',
        parents=['train_model', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_generate_'+PIPELINE_NAME,
        parameter_override={
            'Args/clearml': True,
            'Args/remote': params['remote'],
            'Args/input': params['data_dir'],
            'Args/output': params['emb_dir'],
            'Args/model_path': params['model_path'],
            'Args/s3': params['s3'],
            'Args/s3_dataset_name': params['s3_dataset_name'],
            'Args/exp_name':params['exp_name']
        },
        execution_queue='compute'
    )
    pipe.add_step(name='evaluate',
        parents=['generate_embedding', ],
        base_task_project=PROJECT_NAME,
        base_task_name='pl_evaluate_'+PIPELINE_NAME,
        parameter_override={
            'Args/use_clearml': True,
            'Args/input': params['eval_dir'],
            'Args/emb': params['emb_dir'],
            'Args/label': params['label_path'],
            'Args/model_path': params['model_path'],
            'Args/s3': params['s3'],
            'Args/s3_dataset_name': params['s3_dataset_name'],
            'Args/exp_name':params['exp_name']
        },
        execution_queue='compute'
    )

    
    # pipe.set_default_execution_queue('compute')

    # pipe.add_step(name='generate_embedding', parents=['train_model', ], base_task_project=PROJECT_NAME, base_task_name=args.task_name+' generate')

    # for debugging purposes use local jobs
    # pipe.start_locally(True)

    # Starting the pipeline (in the background)
    if params['remote']:
        pipe.start(queue='cpu-only')
    else:
        pipe.start_locally(True)
    print('done')
