'''
This module holds utility code, that enables participants to interact with
the Global Data Science 3 SageMaker Backend.
Please ensure to modify the constants in the CONFIG area to your data.
'''
import json
import logging
import os
import pathlib
import shutil
import requests

from aws_requests_auth.boto_utils import BotoAWSRequestsAuth
import boto3


##########
# CONFIG #
##########
# This should be changed by the GDSC3 admins.
BASE_HOST = 'z8js7f1x0e.execute-api.eu-west-1.amazonaws.com'

# aws_host needs to contain the FQDN of the api endpoint
# eg: as22asdai2h.execute-api.eu-west-1.amazonaws.com
BOTO3_AUTH = BotoAWSRequestsAuth(
    aws_host=BASE_HOST,
    aws_region='eu-west-1',
    aws_service='execute-api',
)

# This will be provided by the workshop team!
BASE_API_ENDPOINT = 'https://%s/Prod/' % BASE_HOST

# The folders will be ignored while uploading to s3
IGNORED_UPLOAD_FOLDERS = (
    'trained_models',
    '__pycache__',
    'VENV',
    '.VENV',
    '.venv',
)

def retrieve_team_ddn_config_record():
    '''
    Downloads the config from DynamoDB for the given team.
    '''
    full_url = '%s%s' % (
        BASE_API_ENDPOINT,
        'get_team_details/'
    )

    response = requests.get(
        full_url,
        auth=BOTO3_AUTH,
    )

    team_config_dict = json.loads(response.json()['message'])

    logging.info('Downloaded team config: %s from URL: %s' % (
        team_config_dict,
        response.url,
    ))

    return team_config_dict


def get_job_details(job_name):
    '''
    Returns detailed information for a given jobname.
    '''
    full_url = '%s%s' % (
        BASE_API_ENDPOINT,
        'get_job_details/',
    )
    response = requests.get(
        full_url,
        params={
            'job_name': job_name,
        },
        auth=BOTO3_AUTH,
    )
    job_details = json.loads(response.content)['message']
    return job_details


def stop_job(job_name):
    '''
    Stops the job with the given name.
    '''
    full_url = '%s%s' % (
        BASE_API_ENDPOINT,
        'stop_job/',
    )
    response = requests.get(
        full_url,
        params={
            'job_name': job_name,
        },
        auth=BOTO3_AUTH,
    )
    result = response.json()['message']
    return result


def list_all_running_jobs():
    '''
    Lists all running jobs for the configured IAM user.
    '''
    full_url = '%s%s' % (
        BASE_API_ENDPOINT,
        'list_running_jobs/'
    )
    response = requests.get(
        full_url,
        auth=BOTO3_AUTH,
    )
    job_list = response.json()['message']
    return job_list


def download_sagemaker_job_results(job_name):
    '''
    Downloads the results of a given jobname to ../../trained_models/job_name/model.tar.gz
    '''
    # Retrieve current job status and ensure its either stopped or finished.
    job_details = get_job_details(job_name)
    current_status = job_details['TrainingJobStatus']

    if not current_status in ('Completed', 'Stopped'):
        return 'Please try again later, your job is currently in status: %s' % current_status

    s3_output_path = job_details['ModelArtifacts']['S3ModelArtifacts']
    s3_output_region = job_details['HyperParameters']['sagemaker_region'].replace('"', '')

    # Create the local directory
    local_path = get_job_results_folder(job_name)

    if not os.path.exists(local_path):
        os.makedirs(local_path)

    print('Will download training results to %s' % local_path)

    # Extract the bucket name from the Artifact URl.
    bucket_name = s3_output_path.strip('s3://').split('/')[0]
    model_key = s3_output_path.split('%s/' % bucket_name)[-1]

    s3_resource = boto3.resource('s3', region_name=s3_output_region)
    source_bucket = s3_resource.Bucket(bucket_name)

    source_bucket.download_file(
        model_key,
        os.path.join(
            local_path,
            'model.tar.gz'
        ),
    )
    
    return local_path


def extract_results(job_name):
    '''
    Extracts the results of a given jobname to ../../trained_models/job_name/
    '''
    job_results_folder = get_job_results_folder(job_name)
    shutil.unpack_archive(
        os.path.join(job_results_folder, "model.tar.gz"),
        extract_dir=str(job_results_folder),
        format="gztar",
    )


def get_job_results_folder(job_name):
    return os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        '..', '..', 'trained_models', job_name,
    )

def download_sagemaker_job_logs(job_name):
    '''
    Downloads the Cloudwatch Logs for a given job name.
    '''
    # Retrieve the configuration for our team.
    team_config = retrieve_team_ddn_config_record()

    cw_client = boto3.client(
        'logs',
        region_name=team_config['team_region'],
    )

    # Construct the path to the log file.
    log_file_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        '..', '..', 'logs', '%s.txt' % job_name
    )
    print(log_file_path)
    filter_kwargs = {
        'logGroupName': '/aws/sagemaker/TrainingJobs',
        'logStreamNamePrefix': job_name,
        'limit': 10000,
    }

    with open(log_file_path, 'w') as file_handle:
        while True:
            response = cw_client.filter_log_events(**filter_kwargs)

            for event in response['events']:
                file_handle.write(event['message'])
                # Force a newline here, as message might contain newlines as well.
                file_handle.write('\n')
                # Iterate until there is no nextToken in the response anymore.

            try:
                filter_kwargs['nextToken'] = response['nextToken']
            except KeyError:
                break

    return log_file_path


def upload_code_folder_to_s3():
    '''
    This function uploads the local code directory to S3 so that the code is
    available for provisioning the SageMaker process later on.
    '''
    # IMPORTANT
    # THE BASE ASSUMPTION IS, THAT WE WANT TO UPLOAD THE MAIN SRC FOLDER
    # SO WE CONSTRUCT AN ABSOLUTE FILESYSTEM PATH TO THE FOLDER ONE LEVEL
    # ABOVE THE UTILS/ PACKAGE INSIDE SRC/
    src_folder = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        '..'
    )

    # Retrieve the configuration for our team.
    team_config = retrieve_team_ddn_config_record()

    # Retrieve the correct bucket name and region
    target_bucket_name = team_config['team_regional_bucket_name']
    target_region = team_config['team_region']

    # Construct the correct path / prefix for the s3 bucket.
    target_path = '%s/training_code_latest' % team_config['team_user_name']

    # Instantiate the necessary boto3 objects.
    s3_resource = boto3.resource('s3', region_name=target_region)
    target_bucket = s3_resource.Bucket(target_bucket_name)

    # Clean the target_path before uploading new code.
    for key in target_bucket.objects.filter(Prefix=target_path):
        key.delete()

    # Iterate all objects in the source folder.
    # We support one level of subfolders here (for now).
    for filename in os.listdir(src_folder):
        # Ignore blocked sub folders.
        if filename in IGNORED_UPLOAD_FOLDERS:
            continue
        # Check if the current name is a folder.
        full_current_path = os.path.join(src_folder, filename)
        if os.path.isdir(full_current_path):
            print('Found folder: %s' % os.path.join(src_folder, filename))
            # Iterate the subfolder
            for sub_filename in os.listdir(full_current_path):
                if sub_filename in IGNORED_UPLOAD_FOLDERS:
                    continue

                # Construct the correct key for the upload now.
                target_key = '%s/%s/%s' % (
                    target_path,
                    filename,
                    sub_filename,
                )
                full_source_path = os.path.join(
                    src_folder, filename, sub_filename,
                )
                logging.info(
                    'Uploading local file %s to key %s' % (
                        full_source_path,
                        target_key,
                    )
                )
                target_bucket.upload_file(full_source_path, target_key)
        # Handle files in src directory itself
        else:
            target_key = '%s/%s' % (
                target_path,
                filename,
            )
            full_source_path = os.path.join(
                src_folder,
                filename,
            )
            logging.info(
                'Uploading local file %s to key %s' % (
                    full_source_path,
                    target_key,
                )
            )
            target_bucket.upload_file(full_source_path, target_key)

    logging.info('Uploaded code to bucket %s with prefix %s'% (
        target_bucket_name,
        target_path,
    ))


def start_remote_sagemaker_job(base_job_name, entry_point, hyperparams):
    '''
    Starts a SageMaker training job in the assigned region.
    '''
    if len(base_job_name) >= 15:
        raise ValueError('base_job_name must be shorter then 16 chars.')

    if not isinstance(hyperparams, dict):
        raise ValueError('hyperparams must be a dictionary!')

    # Craft the payload.
    payload = {
        'entry_point': entry_point,
        'hyperparameters': hyperparams,
        'base_job_name': base_job_name,
    }

    # Craft the correct endpoint url
    start_job_url = '%s%s' % (
        BASE_API_ENDPOINT,
        'start_job/'
    )

    response = requests.post(
        start_job_url,
        json=payload,
        auth=BOTO3_AUTH
    )

    response = response.json()

    return response['message']
