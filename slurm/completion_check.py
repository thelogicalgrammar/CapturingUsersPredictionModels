"""
A simple script to check what jobs in the original parameter-file
original_parameters.txt finished (i.e. had results written to csv-file).
The script generates a new parameter.txt listing the missing jobs for
a new array job.
"""
import csv


def run_completion_check():
    # 1. Read in what has been completed (as documented in results.csv):
    results = []
    with open('./results/results.csv', 'r') as handle:
        reader = csv.reader(handle)
        # Separate headers from contents
        headers = next(reader)
        for row in reader:
            results.append(row)
    # 1.1 Find indices for parameters of run:
    mod_var_name_idx = headers.index('Modified prior')
    mod_var_value_idx = headers.index('New value of modified prior')
    xo_var_idx = headers.index('X_o')

    # 2. Read in what jobs were planned:
    parameters = []
    with open('./slurm/tmp/original_parameters.txt', 'r') as handle:
        # Just a regular text file:
        reader = csv.reader(handle, delimiter=' ')
        for row in reader:
            parameters.append(row)

    # 3. Compare the two lists
    # Drop everything in results not relevant:
    results = [[row[mod_var_name_idx], row[mod_var_value_idx], row[xo_var_idx]] for row in results]
    uncompleted_jobs = []
    for item in parameters:
        # Match to row in results and remove row from results.
        # If no match, add to uncompleted_jobs.
        try:
            tmp_idx = results.index(item)
            results.pop(tmp_idx)
        except ValueError:
            # Item not in list. Add to jobs to be re-run.
            uncompleted_jobs.append(item)

    # 4. Overwrite old array parameters:
    with open('./slurm/tmp/parameters.txt', 'w') as file:
        for row in uncompleted_jobs:
            file.write("{} {} {}\n".format(row[0], row[1], row[2]))

    if len(uncompleted_jobs) == 0:
        print("All jobs are completed.")
    else:
        print("{} uncompleted jobs".format(len(uncompleted_jobs)))
        print("Parameters for uncompleted jobs stored in ./slurm/tmp/parameters.txt for next array job.")
        print("Consider adding more time to the cluster allocation.")

if __name__ == '__main__':
    # Run actual program
    run_completion_check()
