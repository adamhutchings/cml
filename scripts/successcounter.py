import subprocess
import sys

s = 0
runs = 0
try:
    runs = int(sys.argv[1])
except IndexError:
    runs = 100
except ValueError:
    print(f'Argument {sys.argv[1]} was not a valid number of runs.')
    sys.exit()

for i in range(runs):

    process = subprocess.Popen(["bash", "scripts/drun.sh"], stdout=subprocess.PIPE)
    result = process.communicate()[0]
    if 'Passed' in str(result):
        s += 1

    print(f'Finished run {i + 1} of {runs}.')

print(f'Pass rate: {s} out of {runs} tests.')
