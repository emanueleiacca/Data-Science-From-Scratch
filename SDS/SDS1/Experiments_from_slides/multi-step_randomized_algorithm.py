import random
'''
So the normal algorithm provides the correct answer 99% of the time when the polynomials are not equal.
'''
# Define the polynomials F(x) and G(x) as before
def F(x):
    return (x + 1) * (x - 2) * (x + 3) * (x - 4) * (x + 5) * (x - 6)

def G(x):
    return x**6 - 7 * x**3 + 25

# Single step of the randomized algorithm
def single_step(I):
    r = random.choice(I)
    if F(r) == G(r):
        return True  # Indicates potential equality
    else:
        return False  # Indicates the polynomials are different

# Multi-step randomized algorithm with replacement
def multi_step_with_replacement(I, k):
    '''
    At each step, a new random value 𝑟 is picked independently from the set 𝐼, allowing repeated values.
    the probability of error decreases exponentially as k increases.
    '''
    for _ in range(k):
        if not single_step(I):
            return "F(x) and G(x) are NOT equivalent"
    return "F(x) and G(x) might be equivalent"

# Multi-step randomized algorithm without replacement

def multi_step_without_replacement(I, k):
    '''
    Each time a value r is chosen, it is removed from the set I,
    so it cannot be picked again. This introduces dependency between the runs.
    '''
    sampled_I = random.sample(I, k)  
    for r in sampled_I:
        if F(r) != G(r):
            return "F(x) and G(x) are NOT equivalent"
    return "F(x) and G(x) might be equivalent"

#Honestly the difference between this two methods is often negligible

# Example usage:
I = list(range(1, 10))  # Random values set
k = 5  # Number of iterations
#To improve accuracy, this process can be repeated k times, sampling multiple random values from I.

# Run both algorithms
result_with_replacement = multi_step_with_replacement(I, k)
result_without_replacement = multi_step_without_replacement(I, k)

result_with_replacement, result_without_replacement
