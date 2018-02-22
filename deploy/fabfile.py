
# paramiko logging:
#import logging
#logging.basicConfig(level=logging.DEBUG)

#from fabric.api import run
import fabric
from fabric.api import *

# https://stackoverflow.com/a/43943293/130164
from fabric import network
from fabric.state import connections

def reconnect_current_host():
    network.disconnect_all()
    connections.connect(env.host + ':%s' % env.port)

#from fabric.api import env

env.use_ssh_config=True
#env.use_shell = True
#env.shell = "/bin/sh -c"
#env.disable_known_hosts = True
#env.no_keys = True
#env.use_ssh_config = False

# controlmasters not imported from ssh config
#env.gateway='demeter1'
env.hosts = [
    'infino-run-1.us-east1-b.pici-hammerlab',
    'infino-run-2.us-east1-b.pici-hammerlab',
    'infino-run-3.us-east1-b.pici-hammerlab',
    ]
env.forward_agent=True # so git clone works with private repo

@task
def hello():
	print("hello world")
	local("echo hello")

@task
def host_type():
    #print(run('uname -s', pty=False))
    run('uname -s')

@task    
def diskspace():
    run('df -lh')

@task
def install_docker():
    sudo('apt-get update')
    sudo("""apt-get install -y \
    apt-transport-https \
    ca-certificates \
    build-essential \
    curl \
    software-properties-common \
    tmux""")
    # use convenience script instead (https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/#install-using-the-convenience-script)
    run("curl -fsSL get.docker.com -o get-docker.sh")
    sudo("sh get-docker.sh")
    #sudo("groupadd docker") # already exists
    sudo("usermod -aG docker $USER", shell=False, shell_escape=False, pty=False)
    run("logout") # may have to log back in and out
    reconnect_current_host() # have to flush the connection so that docker group setting gets applied
    #run("su $USER") # might be helpful if above doesn't work

@task
def confirm_docker():
    # confirm docker install
    run("docker run --rm hello-world")

@task
def clone_repo():
    """
    Replace this with either cloning a private repo where you have your data,
    or otherwise getting the data you want to process onto this machine.
    Make sure to have already run the data through `chunker` (see tutorial notebooks) to be ready to hand a chunked file to infino.
    """
    run("ssh-keyscan github.com >> ~/.ssh/known_hosts") # otherwise next line might require a "yes", and "yes | " doesn't seem to work
    # run("git clone git@github.com:hammerlab/immune-infiltrate-explorations.git")
    # with cd("immune-infiltrate-explorations"):
    #     run("git submodule init")
    #     run("git submodule update")
    run("git clone git@github.com:hammerlab/infino.git")
    with cd('infino'):
        run("git checkout develop")
    # clone docker
    run("gcloud docker -- pull gcr.io/pici-hammerlab/infino-env")
    run("docker tag gcr.io/pici-hammerlab/infino-env hammerlab/infino-docker:develop")

@task
def prep_run():
    """
    create output directory
    also compile models
    """
    with cd("infino/models"):
        run("chmod -R 777 $HOME/infino/models")
        run("""
            docker run -it --rm \
            -v $HOME/infino:/home/jovyan/work \
            hammerlab/infino-docker:develop \
            bash -c 'make -C $HOME/cmdstan $HOME/work/models/model6.4_negbinom_matrix_correlation_features_oos_optim_otherbucket'
            """)
        # TODO
        #run("make -C $HOME/cmdstan model6.5_negbinom_matrix_correlation_features_oos_optim_otherbucket_samplevariance")
        run("mkdir -p ../out")
        run("chmod -R 777 ../out")

@task
def blank_run():
    """
    confirm that everything works
    """
    with cd("infino"):
        run("""docker run -it --rm \
            -v $HOME/infino:/home/jovyan/work \
            hammerlab/infino-docker:develop \
            bash -c 'cd work/ && which execute-model && python -c "print(0)"'
            """)
        # this is the preferred way to run: with `docker run -d` and NOT `python -i`
        run("""docker run -it -d --rm \
            -v $HOME/infino:/home/jovyan/work \
            hammerlab/infino-docker:develop \
            bash -c 'cd work/ && which execute-model && python -c "print(0)" 2>&1 | tee out/test_remote_run.txt'
            """)
        run("sleep 2 && cat out/test_remote_run.txt")
        print("expected to see 'out: 0' above")

@task
def bootstrap_node():
    """
    run this with: fab bootstrap_node
    """
    install_docker() # install docker
    confirm_docker() # make sure works from our user
    clone_repo() # clone repo
    prep_run() # prepare logs dir, set permissions
    blank_run() # test our installation


# @task
# def test_tmux():
#     # https://stackoverflow.com/a/20768173/130164
#     run("tmux new -d -s foo")
#     run("tmux send -t foo.0 ls ENTER")
#     # may be better to do `-d` in docker run


@task
def test_per_host():
    print(env.host_string)
    print(env.host)


from host_task_map import map_host_to_task, map_host_to_docker_name


@task
def run_infino(dryrun='1'):
    """
    defaults to dry run
    """
    print(env.host_string)
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    shell_script = map_host_to_task[env.host_string]
    with cd("infino"):
        run("pwd")
        if dryrun == '0': # have to explicitly pass a 0 to activate the real run. note all args are strings with fabfile
            # run for real
            run(shell_script)
        else: 
            # dry run
            print("Would run:")
            print(shell_script)



@task
def clean_up_docker_containers(dryrun='1'):
    """
    use only if something went wrong and need to destroy the docker containers
    """
    print(env.host_string)
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    if dryrun == '0':
        shell_script = 'docker rm %s' % map_host_to_docker_name[env.host_string]
        print(shell_script)
        run(shell_script)

@task
def docker_logs():
    print(env.host_string)
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    shell_script = 'docker logs %s' % map_host_to_docker_name[env.host_string]
    print(shell_script)
    run(shell_script)


@task
def docker_stop(dryrun='1'):
    # have to run with dryrun=0
    print(env.host_string)
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    shell_script = 'docker stop %s' % map_host_to_docker_name[env.host_string]
    print(shell_script)
    if dryrun == '0':
        run(shell_script)

@task
def docker_kill(dryrun='1'):
    # have to run with dryrun=0
    print(env.host_string)
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    shell_script = 'docker kill %s' % map_host_to_docker_name[env.host_string]
    print(shell_script)
    if dryrun == '0':
        run(shell_script)

@task
def stansummary(dryrun='1'):
    print(env.host_string)
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    slug = map_host_to_docker_name[env.host_string]
    #logs/tcgakirc_cut1_tpm_raw/sampling_log.cohort.tcgakirc_cut1_tpm_raw.txt_0.csv \\

    does_file_exist = fabric.contrib.files.exists('$HOME/immune-infiltrate-explorations/model-single-origin-samples/infino-rcc/logs/{slug}/stansummary.{slug}.csv'.format(slug=slug))
    assert fabric.contrib.files.exists('$HOME/get-docker.sh') # should exist. this is a test of $HOME being completed.

    shell_script = """docker run -it -d --name stansum_{slug} \\
-v $HOME/immune-infiltrate-explorations:/home/jovyan/work \\
hammerlab/infino-docker:latest \\
bash -c "cd work/model-single-origin-samples/infino-rcc/ && stansummary --csv_file=logs/{slug}/stansummary.{slug}.csv \\
logs/{slug}/sampling_log.cohort.{slug}.txt_0.csv \\
logs/{slug}/sampling_log.cohort.{slug}.txt_1.csv \\
logs/{slug}/sampling_log.cohort.{slug}.txt_2.csv \\
logs/{slug}/sampling_log.cohort.{slug}.txt_3.csv > /dev/null;"
""".format(slug=slug)
    if dryrun == '0': # have to explicitly pass a 0 to activate the real run. note all args are strings with fabfile
        # run for real
        print(slug)
        if does_file_exist:
            print('file exists, didnt run')
            return
        run(shell_script)
    else: 
        # dry run
        print('dry run', slug)
        print('does file already exist on remote: ', does_file_exist)
        print(shell_script)
        print()


@task
def docker_container_exit_code():
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    slug = map_host_to_docker_name[env.host_string]
    print(env.host_string, slug)
    container_name = slug # 'stansum_%s' % slug
    run("docker inspect --format='{{.State.ExitCode}}' %s" % container_name)

@task
def stansummary_docker_status():
    if not env.host_string in map_host_to_task.keys():
        print("MISSING!")
        return
    slug = map_host_to_docker_name[env.host_string]
    print(env.host_string, slug)
    container_name = 'stansum_%s' % slug
    run("docker inspect --format='{{.State.ExitCode}}' %s" % container_name)

@task
@parallel
def upload_to_bucket(dryrun='1'):
    slug = map_host_to_docker_name[env.host_string]
    print(env.host_string, slug)
    # -n for dry run
    #shell_script ="gsutil -m rsync -n immune-infiltrate-explorations/model-single-origin-samples/infino-rcc/logs/{slug}/ gs://infino-hammerlab/{slug}/".format(slug=slug)
    shell_script ="gsutil -m rsync infino/out/ gs://infino-hammerlab/run_output/{slug}/".format(slug=slug)
    if dryrun == '0':
        run(shell_script)
    else:
        print(shell_script)

# run with: fab host_type


"""
see this for how to modify per host:
https://stackoverflow.com/a/24781166/130164 !!!!
https://stackoverflow.com/a/11603028/130164

"""