'''
This module demonstrates how to list your currently running remote jobs.
'''
from utils.remote_sagemaker import list_all_running_jobs


if __name__ == '__main__':
    job_list = list_all_running_jobs()
    print(job_list)