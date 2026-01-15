def epsilon_step_by_step(step, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=10000):
    return epsilon_end + (epsilon_start - epsilon_end) * max(0,(epsilon_decay - step)) / epsilon_decay