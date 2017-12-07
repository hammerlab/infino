import subprocess

commands = [
    'echo hi',
    'echo to stdout; echo to stderr 1>&2; exit 1',
    'sleep 3 && exit 1',
    'sleep 4'
]

def flush_stream_to_logs(proc, stdout_log, process_id): #, stderr_log):
    pipe_data = proc.communicate()
    print(process_id, ":", pipe_data)
    stdout_log.write(str(pipe_data) + '\n')
    # for data, log in zip(pipe_data, stdout_log): #(stdout_log, stderr_log)):
    #     # Add whatever extra text you want on each logged message here
    #     log.write(str(data) + '\n')

# for c in commands:
#     print(c)
#     try:
#         completed = subprocess.run(
#             c,
#             check=True,
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT, # PIPE
#             universal_newlines=True,
#         )
#     except subprocess.CalledProcessError as err:
#         print('ERROR:', err)
#     else:
#         print('returncode:', completed.returncode)
#         # print('Have {} bytes in stdout: {!r}'.format(
#         #     len(completed.stdout),
#         #     completed.stdout.decode('utf-8'))
#         # )

procs=[]
for c in commands:
    print(c)
    proc = subprocess.Popen(
        c,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    procs.append(proc)

with open('stdout.test', 'w') as stdout_log:
    while any(proc.returncode is None for proc in procs):
        for process_id, proc in enumerate(procs):
            try:
                flush_stream_to_logs(proc, stdout_log, process_id)
            except Exception as e:
                pass
    
    # after finished
    for process_id, proc in enumerate(procs):
        try:
            flush_stream_to_logs(proc, stdout_log, process_id)
        except Exception as e:
            pass
        print(process_id, 'returncode:', proc.returncode)

print('done')

