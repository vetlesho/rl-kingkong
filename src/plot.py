import re
import sys
import matplotlib.pyplot as plt

# pulls the step and avg(100) out of each log line
pattern = re.compile(r"step\s+(\d+).*avg\(100\)\s+(-?[\d.]+)")

for path in sys.argv[1:]:
    steps, averages = [], []
    with open(path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                averages.append(float(match.group(2)))
    plt.plot(steps, averages, label=path)

plt.xlabel("Step")
plt.ylabel("Average score over last 100 games")
plt.legend()
plt.grid(True)
plt.show()