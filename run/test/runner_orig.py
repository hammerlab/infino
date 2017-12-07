import subprocess

commands = [
    'echo hi',
    'echo to stdout; echo to stderr 1>&2; exit 1',
]

for c in commands:
    print(c)
    try:
        completed = subprocess.run(
            c,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # PIPE
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as err:
        print('ERROR:', err)
    else:
        print('returncode:', completed.returncode)
        # print('Have {} bytes in stdout: {!r}'.format(
        #     len(completed.stdout),
        #     completed.stdout.decode('utf-8'))
        # )

print('done')

