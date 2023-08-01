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

pbarlen = 50
efruns = runs + pbarlen - (runs % pbarlen)
if runs % pbarlen == 0:
    efruns = runs

print(f'Progress:\n|{"-" * pbarlen}|')
print(f'|', end='', flush=True)

ps = 0

for i in range(runs):

    process = subprocess.Popen(["bash", "scripts/drun.sh"], stdout=subprocess.PIPE)
    result = process.communicate()[0]
    if 'Passed' in str(result):
        s += 1

    if (s - 1) % (efruns // pbarlen) == 0:
        print(f'-', end='', flush=True)
        ps += 1

print(f'{"-" * (pbarlen - ps)}|')
print(f'Pass rate: {s} out of {runs} tests.')
