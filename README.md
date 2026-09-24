# My First Simulated Neuron - A README of Everything I Learned

This is my personal documentation of building my very first Leaky Integrate-and-Fire (LIF) neuron simulation in Python. I used AI to help me understand these concepts, code and to format this READMe and make it more easier to read!!! This is for my personal enrichment so do not come at me for using AI. First, we must learn about the science behind a neuron and then I will explain the code.

If you still don't understand go to this link --->  https://youtube.com/watch?v=rSExvwCVRYg

---

## Part 1: The Science Behind the Neuron

### The Leaky Integrate-and-Fire Model

The LIF model is the foundational building block of computational neuroscience. It simplifies a biological neuron into a few core physical properties.

A useful analogy to visualize this model is a **bucket with a small hole in the bottom**:

- **The bucket** represents the neuron's membrane, which stores electrical charge.
- **Water** represents the membrane potential (voltage).
- **The hole** represents the "leak" of ions through the membrane.
- **Pouring water** represents the input current injected into the neuron.
- **The top rim of the bucket** represents the firing threshold.

### The Resting Potential and Threshold

In a resting state, the inside of a neuron is negatively charged compared to the outside. This is measured as the **Resting Membrane Potential**, typically around **-70 mV**.

- The negative sign indicates that the inside is 70 millivolts *more negative* than the outside environment (which is defined as 0 mV).
- This electrical difference is maintained by ion pumps that push positive Sodium (Na+) ions out and keep negative ions inside.

For the neuron to send a signal, it must become *less negative*. It needs to **depolarize** until it reaches the **Spike Initiation Threshold** (typically around **-50 mV**).

### The Action Potential (The Spike)

When the membrane potential reaches the threshold, a rapid chain reaction occurs:

1.  **Depolarization (The Upstroke)**: Voltage-gated Sodium (Na+) channels open. Because there is a high concentration of Na+ outside and a negative charge inside, Na+ ions rush into the cell. This floods the neuron with positive charge, driving the membrane potential rapidly up to approximately **+40 mV** (the peak of the action potential).
2.  **Repolarization (The Downstroke)**: The Sodium channels quickly inactivate (close). Simultaneously, voltage-gated Potassium (K+) channels open. K+ ions are crowded inside, so they rush *out* of the cell, dragging the voltage back down.
3.  **The Undershoot (Afterhyperpolarization)**: The K+ channels are slow to close, causing the voltage to drop slightly *below* the resting potential, to roughly **-80 mV**.

### The Refractory Period

Immediately after firing, the neuron enters an **Absolute Refractory Period** (lasting about 1-2 ms). During this time:

- The Sodium channels are physically inactivated and cannot re-open, regardless of the input strength.
- The neuron is effectively "dead" to new stimuli and cannot fire another action potential.
- This mechanism ensures the signal travels in one direction and sets a maximum firing frequency for the neuron.

### The Mathematical Model (The Physics)

The subthreshold dynamics of the neuron are governed by the physics of an **RC circuit** (a Resistor and Capacitor in parallel).

- The membrane acts as a **Capacitor (C)** that stores charge.
- The ion channels act as a **Resistor (R)** that allows charge to leak out.

According to Kirchhoff's Current Law, the injected current (\( I \)) must split between charging the capacitor and leaking through the resistor:

\[
I = C_m \frac{dV}{dt} + g_L (V - V_{rest})
\]

Where:
- \( C_m \) is the membrane capacitance.
- \( g_L \) is the leak conductance (the inverse of resistance).
- \( V \) is the current membrane potential.
- \( V_{rest} \) is the resting potential.

Rearranging the equation to solve for the change in voltage over time gives us the **LIF differential equation**:

\[
\tau \frac{dV}{dt} = -(V - V_{rest}) + R_m I
\]

In this equation, \( \tau \) (the **Membrane Time Constant**) is defined as \( \tau = \frac{C_m}{g_L} \), and \( R_m \) is the membrane resistance (\( \frac{1}{g_L} \)).

The membrane time constant (\( \tau \)) determines how quickly the neuron responds to inputs. A larger \( \tau \) means the neuron integrates signals over a longer time window (less leaky), while a smaller \( \tau \) makes it leakier and more forgetful.

---

## Part 2: The Code Implementation

The computer cannot solve continuous differential equations. Therefore, the simulation uses the **Forward Euler Method**. This breaks continuous time into tiny, discrete steps (`dt`), and calculates the change in voltage at each step.

Here is the Python code, broken down into its core components:

```python
import numpy as np
import matplotlib.pyplot as plt
```

### 1. The Simulation Clock
```python
dt = 0.1       # Time step size (ms). Smaller values increase resolution.
T = 200        # Total simulation time (ms).
time = np.arange(0, T, dt)
```
This creates an array of discrete timestamps representing each "frame" of the simulation. In this case, we take a snapshot every 0.1 milliseconds for 200 milliseconds.

### 2. Neuron Parameters
```python
V_rest = -70   # Resting membrane potential (mV).
V_reset = -80  # Afterhyperpolarization level (mV).
V_peak = 40    # Peak of the action potential (mV).
threshold = -50 # Spike initiation threshold (mV).
```
These constants define the biophysical "rules" of the neuron. `V_reset` simulates the Potassium-driven overshoot, and `V_peak` represents the peak Sodium-driven depolarization.

### 3. Input and Passive Properties
```python
I = 25         # Injected current (scaled to mV in this model).
tau = 20       # Membrane time constant (ms).
```
`I` is the driving force trying to push the neuron toward threshold. `tau` controls the speed of the voltage changes.

### 4. The Refractory Mechanism
```python
refractory_period = 2.0
refractory_steps = int(refractory_period / dt) # 2.0 / 0.1 = 20 steps
refractory_timer = 0
```
Instead of complex biology, the code uses a countdown timer. After a spike, the timer is set to `20` (representing 2ms). While the timer is active, the voltage is forced to stay at `V_reset` and the neuron ignores all incoming current.

### 5. The Simulation Loop (Forward Euler Integration)
```python
V = np.zeros(len(time))
V[0] = V_rest

for i in range(1, len(time)):

    # --- Refractory Check ---
    if refractory_timer > 0:
        V[i] = V_reset
        refractory_timer -= 1
        continue  # Skip the math; neuron is recovering.

    # --- Leaky Integration (The RC Circuit) ---
    dV = (-(V[i-1] - V_rest) + I) * dt / tau
    V[i] = V[i-1] + dV

    # --- Spike Generation & Reset ---
    if V[i] >= threshold:
        V[i] = V_peak                # Draw the spike upstroke.
        refractory_timer = refractory_steps  # Activate the 2ms time-out.
```

**Step-by-step logic of the loop:**
1.  **Refractory Check**: If the timer is active, force the voltage to `V_reset`, decrement the timer, and skip the integration math.
2.  **Leaky Integration**: Calculate the change in voltage based on the RC circuit equation:  
    `dV = (-(V - V_rest) + I) * dt / tau`  
    This computes the new voltage for the current time step.
3.  **Spike Generation**: If the integrated voltage meets or exceeds the `threshold`, immediately overwrite the voltage to `V_peak` (the spike), and start the refractory timer. On the *next* iteration, the timer will drop the voltage down to `V_reset`, simulating the downstroke and recovery.

### 6. Visualization
```python
plt.plot(time, V)
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.title("LIF Neuron with Refractory Period")
plt.grid(True)
plt.show()
```
This renders the final array of voltage values against time, producing the characteristic "sawtooth" pattern of an integrating and firing neuron.

---

## Final Summary

The neuron is treated as a passive RC circuit that constantly leaks charge. An external input drives the membrane potential upward. If the potential reaches the threshold before the "leak" drains it away, the model triggers an all-or-nothing action potential event and forces a hard reset. The refractory period enforces a biological limitation on how frequently the neuron can fire. This simple mathematical construct forms the foundation of many modern neural network and brain simulation models.