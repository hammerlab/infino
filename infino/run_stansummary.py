import argparse
import subprocess


def stansummary(output_fname, input_names):
    input_names_concat = ' '.join(input_names)
    command = """echo stansummary --csv_file={output_fname} {input_names_concat} > /dev/null;""".format(
        output_fname=output_fname,
        input_names_concat=input_names_concat
    )
    print(command)
    try:
        completed = subprocess.run(
            command,
            #check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as err:
        print('ERROR:', err)
        return
    if completed.returncode != 0:
        print('return code:', completed.returncode)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_log', action='append', required=True, help="sampling log filename(s)")
    parser.add_argument('--output_file', required=True, help='output file name')
    parser.add_argument('--exclude_broken_chains', action='store_true', help="detect and exclude broken chains")
    
    args = parser.parse_args()
    
    # TODO: implement exclude_broken_chains
    
    stansummary(args.output_file, args.sample_log)

if __name__ == '__main__':
    main()