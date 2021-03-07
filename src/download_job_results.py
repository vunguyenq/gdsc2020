'''
This module can be used to download and extract the results of the training job for a given job name
'''
import sys
from pprint import pprint

from utils.remote_sagemaker import download_sagemaker_job_results
from utils.remote_sagemaker import extract_results


if __name__ == '__main__':
    # Check if sys.argv contains an argument / jobname
    if len(sys.argv) < 2:
        print(
            'Please invoke the script like this:\n$ python download_job_results.py ExampleJob-Team-2020-03-06-08-24-55-734\n'
            'The name of your job needs to be the first an only argument.'
        )

    else:
        job_name = sys.argv[1]
        print('Will download results for job name: %s' % job_name)
        result = download_sagemaker_job_results(job_name)
        print('Will extract the results to %s' % result)
        extract_results(job_name)
        print('You can now inspect your results here: %s' % result)
