'''
This module is an example for how to trigger a remote training job on AWS SageMaker.

Please ensure that your the TEAM_NAME constant in utils/remotes_sagemaker.py is adjusted
and reflects your actual teamname.
'''
from utils.remote_sagemaker import retrieve_team_ddn_config_record
from utils.remote_sagemaker import upload_code_folder_to_s3
from utils.remote_sagemaker import start_remote_sagemaker_job


if __name__ == '__main__':
    # Retrieve your teams configuration (Region, Username, etc.) from the API.
    # Uncomment these lines to better understand, what is happening here.
    # team_config = retrieve_team_ddn_config_record()
    # print('Retrieved team_config!')
    # pprint(team_config)

    # Upload your current sourcecode to S3.
    # This step is a necessary preperation for starting the remote job.
    # The folder, containing this very file here, 'src/' within the repo,
    # will be uploaded to your teams code path.
    upload_code_folder_to_s3()

    # Now start the actual training job.
    job_info = start_remote_sagemaker_job(
        base_job_name='FirstTraining',
        # This MUST point to a file, relative to the src/ folder.
        # In this very example we use the provided local training script.
        entry_point='local_training_siamese_mobilenet_from_images.py',
        #entry_point='logging_test.py',
        # Tweak your hyperparams here.
        hyperparams={
            'epochs': 250,
            'learning_rate': 0.0001, 
            'batch_size': 128,
            'steps_per_epoch': 90
        },
    )
    print('Started job with name: %s' % job_info)
  