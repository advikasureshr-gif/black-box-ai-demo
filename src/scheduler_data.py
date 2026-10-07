import numpy as np
import pandas as pd


FEATURES = [
    "cpu_required",
    "memory_required",
    "priority",
    "deadline",
    "queue_length",
]

VM_NAMES = {
    0: "VM-1",
    1: "VM-2",
    2: "VM-3",
    3: "VM-4",
}


def generate_scheduling_decisions(tasks):
    """
    Deterministic black-box scheduling policy.

    Returns:
        numpy array containing VM names.
    """

    decisions = []

    for _, task in tasks.iterrows():

        cpu = task["cpu_required"]
        memory = task["memory_required"]
        priority = task["priority"]
        deadline = task["deadline"]
        queue = task["queue_length"]

        if priority >= 8 and deadline <= 100:
            vm = 0 if cpu <= 8 else 2

        elif cpu >= 12 or memory >= 24:
            vm = 2 if queue <= 8 else 3

        elif cpu >= 7 or memory >= 12:
            vm = 1 if deadline <= 120 else 2

        else:
            vm = 0 if queue <= 5 else 1

        decisions.append(VM_NAMES[vm])

    return np.asarray(decisions)


def generate_tasks(n=1000, random_state=42):

    rng = np.random.default_rng(random_state)

    tasks = pd.DataFrame({
        "cpu_required": rng.integers(1, 17, n),
        "memory_required": rng.integers(1, 33, n),
        "priority": rng.integers(1, 11, n),
        "deadline": rng.integers(20, 301, n),
        "queue_length": rng.integers(0, 21, n),
    })

    tasks["assigned_vm"] = generate_scheduling_decisions(tasks)

    return tasks