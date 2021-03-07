'''
This module can be used to monitor a remote job and its status changes.
'''
import datetime
import os
import sys
import time

from utils.remote_sagemaker import get_job_details

def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


if __name__ == '__main__':
    # Check if sys.argv contains an argument / jobname
    if len(sys.argv) < 2:
        print(
            'Please invoke the script like this:\n$ python observe_job_status.py ExampleJob-Team-2020-03-06-08-24-55-734\n'
            'The name of your job needs to be the first an only argument.'
        )

    else:
        job_name = sys.argv[1]
        
        while True:
            job_info = get_job_details(job_name)
            clear_screen()
            print('#'*100)
            print('### Job Name: %s' % job_info['TrainingJobName'])
            print('### Job CreatedAt: %s' % job_info['CreationTime'])
            print('#'*100)
            print('### Job Primary Status: %s' % job_info['TrainingJobStatus'])
            print('#'*100)
            print('### Job Detail Status: %s' % job_info['SecondaryStatus'])
            print('### Job Status transitions:')
            for transition in job_info['SecondaryStatusTransitions']:
                print('###\t%s\t%s\t%s' % (
                    transition['StartTime'],
                    transition['Status'],
                    transition['StatusMessage'],
                ))
            print('#'*100)
            print(
                'This overview was last updated at: %s' % datetime.datetime.now().strftime('%H:%M:%S')
            )
            print(
                'To stop monitoring this job please use ctrl+c'
            )
            print('#'*100)
            time.sleep(12)
