# Dynamic-MDP
# Designing an Optimal Policy for UAVs in a Dynamic MDP Environment to Prevent Birdstrikes

## Introduction
In this project, I’ve designed and implemented a **Dynamic Markov Decision Process (MDP)** to guide an **Unmanned Aerial Vehicle (UAV)** through a **3D environment** filled with moving obstacles—specifically, birds. The goal is to navigate the UAV to a target location while avoiding collisions, using a model-based approach called **Policy Iteration**.

To make this solution tangible and visually engaging, I’ve also built a **visualization tool using Pygame**. This document walks you through the problem, the solution, and how to run the code.

## Problem Overview
Imagine a UAV flying through a dynamic environment where birds move unpredictably. The objective is simple yet challenging: **guide the UAV from its starting position to the goal while avoiding collisions**.

- The UAV operates in a **20×20×5 grid** (a 3D state space).
- The UAV must avoid **up to 100 moving birds** with **linear velocity** and **observation noise (0.1 grid size in position & velocity)**.
- The goal state is any position where `x = 19` (far end of the grid).

## Solution Approach
This problem is formulated as a **Dynamic MDP**, where the UAV makes optimal decisions based on its current state and the predicted positions of birds.

### Step 1: Define the State Space & Goal
- **State Space:**  `(x, y, z)`, where `x, y, z` represent the UAV's position.
- **Goal State:** Any state where `x = 19`.
- **Starting Point:** The UAV starts at `(0, y, z)`.

### Step 2: Model Bird Movement
- Each bird has a **constant velocity vector**.
- Bird positions are **predicted** with a **small observation error (0.1 grid units in position & velocity)**.
- This uncertainty forces the UAV to adapt dynamically.

### Step 3: UAV Decision-Making with Policy Iteration
#### **Action Space:**
The UAV can take **five actions**:
- **F** (Forward)
- **L** (Left)
- **R** (Right)
- **U** (Up)
- **D** (Down)

#### **Transition Model:**
- The UAV moves with a probability of **0.98** to the intended state and **0.01** to adjacent states.
- Probabilistic transitions reflect real-world uncertainties in UAV control.

#### **Reward Function:**
- **+100** for reaching the goal.
- **-100** for colliding with a predicted bird location.
- **-1 to -7 step penalties** to encourage efficient movement.

#### **Policy Iteration Algorithm:**
1. **Policy Evaluation:** Compute state values using a discount factor of **0.99**.
2. **Policy Improvement:** Update the policy by choosing the best action.
3. **Repeat until convergence.**

### Step 4: Visualization (Pygame)
To demonstrate the UAV's decision-making in real-time:
- The **UAV** is **green**.
- The **birds** are **red**.
- The **grid** is **gray/white**.
- The simulation updates every second, showing UAV movement and collision avoidance.

## Alternative UI Option
I have also included **another UI**, which you can use instead of the default visualization. However, if you choose to use it, **you must integrate it with the existing logic** to ensure consistency in UAV decision-making.

## How to Clone and Run
### 1. Clone the Repository:
```bash
git clone https://github.com/ishapatel97/Dynamic-MDP.git
cd Dynamic-MDP
```

### 2. Install Dependencies:
```bash
pip install pygame
```

### 3. Run the Visualization:
```bash
python Visualization.py
```

### 4. (Optional) Use Alternative UI:
Modify the UI script and integrate it with `Dynamic_MDP_test.py`.

## Results & Evaluation
I tested the UAV's policy with varying numbers of birds (10 to 100). The UAV learned different avoidance strategies based on bird distribution:
- **Few birds:** UAV moves primarily **Forward & Down**.
- **More birds at the same height:** UAV uses **Forward & Left/Right** to navigate gaps.
- **Dense 3D bird distribution:** UAV applies a **mix of all actions** to find a clear path.

## Conclusion
This project demonstrates how **Dynamic MDPs with Policy Iteration** can enable UAVs to navigate uncertain environments with moving obstacles. Future improvements could include:
- **Non-linear bird motion models**.
- **More realistic sensor noise simulation**.
- **Real-world UAV hardware integration**.

Contributions and feedback are welcome! 🚀

