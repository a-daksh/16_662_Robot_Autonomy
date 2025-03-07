import numpy as np
import matplotlib.pyplot as plt

file_groups = [
    ['force_vs_time_force_control.csv', 'force_vs_time_impedance_control.csv'],
    ['moving_force_vs_time_force_control.csv', 'moving_force_vs_time_impedance_control.csv']
]

titles = [
    "Force vs Time (Force Control vs Impedance Control)",
    "Moving Force vs Time (Force Control vs Impedance Control)"
]

colors = ['blue', 'red']  
labels = ['Force Control', 'Impedance Control']

for i, (files, title) in enumerate(zip(file_groups, titles)):
    plt.figure(i + 1)
    
    for j, file_name in enumerate(files):
        data = np.loadtxt(file_name, delimiter=',')
        time = data[:, 0]  # Time 
        force = data[:, 1]  # Force 
        plt.plot(time, force, label=labels[j], color=colors[j])

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Force (Newtons)")
    plt.axhline(y=15, linewidth=1, color='k', linestyle='--', label="Reference Force (15N)")  # Reference line
    plt.legend()
    plt.grid(True)

plt.show()
