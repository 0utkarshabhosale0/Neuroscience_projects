import numpy as np
import matplotlib.pyplot as plt

# Simulation settings
dt = 0.1  
T = 200   
time = np.arange(0, T, dt)

# neuron properties
V_rest = -70       
V_reset = -80    
V_peak = 40        
threshold = -50    

#starting voltage
V = np.zeros(len(time))
V[0] = V_rest

#input current
I = 25             

# membrane time constant
tau = 20

#Refractory period timer (stops it from firing again instantly)
refractory_period = 2.0  # ms
refractory_steps = int(refractory_period / dt)  # how many calculation steps = 2ms
refractory_timer = 0     # starts at 0 (not in refractory period)

# Simulate the neuron
for i in range(1, len(time)):

    # The Refractory Period
    # If the timer is counting down, the neuron is "recovering".
    # Force the voltage to stay at the reset level and do NOT calculate anything else.
    if refractory_timer > 0:
        V[i] = V_reset
        refractory_timer -= 1
        continue  

    # Normal Leaky Integration
    dV = (-(V[i-1] - V_rest) + I) * dt / tau
    V[i] = V[i-1] + dV

    # The Fire
    # If we hit the threshold, blast the voltage up to +40mV
    if V[i] >= threshold:
        V[i] = V_peak   # Draw the lightning bolt straight up

        # The Reset
        refractory_timer = refractory_steps

# Plot the result
plt.plot(time, V)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("Leaky Integrate and Fire (LIF) neuron")
plt.grid(True)
plt.show()