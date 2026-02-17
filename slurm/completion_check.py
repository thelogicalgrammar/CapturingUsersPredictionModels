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
    #results = [[row[mod_var_name_idx], float(row[mod_var_value_idx]), row[xo_var_idx]] for row in results]
    # results_xo = [row[xo_var_idx] for row in results]
    results = [[row[mod_var_name_idx], float(row[mod_var_value_idx]), row[xo_var_idx]] for row in results]
    uncompleted_jobs = []
    for i, item in enumerate(parameters):
        # Match to row in results and remove row from results.
        # If no match, add to uncompleted_jobs.
        try:
            # Need to not match on "old" X_o if we want to accept "old" results.
            # tmp_idx = results.index([item[0], float(item[1]), item[2]])  # Need to match on float-value rather than string.. FIX.
            tmp_idx = results.index([item[0], float(item[1]), row[2]])  # Need to match on float-value rather than string.. FIX.
            results.pop(tmp_idx)
            # results_xo.pop(tmp_idx)
        except ValueError:
            # Item not in list. Add to jobs to be re-run.
            uncompleted_jobs.append(item)

    # 4. Overwrite old array parameters:
    with open('./slurm/tmp/parameters.txt', 'w') as file:
        for row in uncompleted_jobs:
            file.write("{} {:.6f} {}\n".format(row[0], float(row[1]), row[2]))

    if len(uncompleted_jobs) == 0:
        print("All jobs are completed.")
    else:
        print("{} uncompleted jobs".format(len(uncompleted_jobs)))
        print("Parameters for uncompleted jobs stored in ./slurm/tmp/parameters.txt for next array job.")
        print("Consider adding more time to the cluster allocation.")


def summarize_results(result_file='./results/results.csv'):
    """
    Function that counts how many lines there are in
    result_file for every prior parameter and value.
    
    result_file (str): Filename and path of csv file to summarize.
    """
    results = []
    with open(result_file, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        for line in reader:
            results.append(line)

    # Find unique variable names (idx 4 for name, idx 5 for value)
    var_names = list(set([item[4] for item in results]))
    results_by_var_name = {}
    for var_name in var_names:
        tmp = [line[5] for line in results if line[4] == var_name]
        unique_values = list(set(tmp))
        unique_values = [float(item) for item in unique_values]
        unique_values.sort()
        related_counts = {}  # Reset
        for value in unique_values:
            value_count = sum([1 for item in tmp if float(item) == value])
            related_counts[str(value)] = value_count
        results_by_var_name[str(var_name)] = related_counts

    for key in results_by_var_name.keys():
        print("=" * 40)
        print(key)
        print("=" * 40)
        for key_2 in results_by_var_name[key]:
            print("Count for {}: {}".format(key_2, results_by_var_name[key][key_2]))


if __name__ == '__main__':
    # Run actual program
    run_completion_check()
