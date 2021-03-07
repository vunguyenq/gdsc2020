'''
This module can be used to retrieve a detailed summary for a given job name
'''
import sys
from pprint import pprint

from utils.remote_sagemaker import get_job_details


if __name__ == '__main__':
    # Check if sys.argv contains an argument / jobname
    if len(sys.argv) < 2:
        print(
            'Please invoke the script like this:\n$ python get_job_details.py ExampleJob-Team-2020-03-06-08-24-55-734\n'
            'The name of your job needs to be the first an only argument.'
        )

    else:
        job_name = sys.argv[1]
        print('Will retrieve infos for job name: %s' % job_name)
        job_info = get_job_details(job_name)
        pprint(job_info)