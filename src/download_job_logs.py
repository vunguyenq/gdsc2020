'''
This module can be used to retrieve the CloudWatch logs for a given job name.
'''
import sys

from utils.remote_sagemaker import download_sagemaker_job_logs


if __name__ == '__main__':
    # Check if sys.argv contains an argument / jobname
    if len(sys.argv) < 2:
        print(
            'Please invoke the script like this:\n$ python download_sagemaker_job_logs.py ExampleJob-Team-2020-03-06-08-24-55-734\n'
            'The name of your job needs to be the first an only argument.'
        )

    else:
        job_name = sys.argv[1]
        print('Will download logs for job name: %s' % job_name)
        result = download_sagemaker_job_logs(job_name)
        print('You can now open the log file here: %s' % result)
