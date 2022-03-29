from clearml import Task
from clearml.automation import PipelineController
import argparse
import experiment

PROJECT_NAME = 'facenet'

if __name__ == '__main__':
    # Connecting ClearML with the current pipeline,
    # from here on everything is logged automatically


    # parser = argparse.ArgumentParser()
    # parser = experiment.Experiment.add_experiment_args()
    # args = parser.parse_args()

    # exp = experiment.Experiment(args)

    pipe = PipelineController(
        name='Pipeline_test',
        project=PROJECT_NAME,
        version='0.0.1',
        add_pipeline_tags=False,
    )
    print("Pipe created")
    pipe.add_step(name='train_model',
        base_task_project=PROJECT_NAME,
        base_task_name='pl_train',
        parameter_override={'Args/batch_size': 32}
    )
    
    pipe.set_default_execution_queue('default')
    
    # exp.run_experiment()
    # pipe.add_step(name='generate_embedding', parents=['train_model', ], base_task_project=PROJECT_NAME, base_task_name=args.task_name+' generate')

    # for debugging purposes use local jobs
    pipe.start_locally(True)

    # Starting the pipeline (in the background)
    # pipe.start()

    print('done')
