import subprocess

s = 0

for i in range(1000):

    process = subprocess.Popen(["bash", "scripts/drun.sh"], stdout=subprocess.PIPE)
    result = process.communicate()[0]
    if 'Passed' in str(result):
        s += 1

    print(f'Finished run {i + 1} of {s}.')

print(f'Success rate: {round(s/10, 2)}%.')
