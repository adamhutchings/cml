import subprocess

s = 0
runs = 100

for i in range(runs):

    process = subprocess.Popen(["bash", "scripts/drun.sh"], stdout=subprocess.PIPE)
    result = process.communicate()[0]
    if 'Passed' in str(result):
        s += 1

    print(f'Finished run {i + 1} of {runs}.')

print(f'Success rate: {round(s/(runs/100), 2)}%.')
