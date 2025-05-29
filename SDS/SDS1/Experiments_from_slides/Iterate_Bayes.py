import random

# Define the polynomials F(x) and G(x) as before
def F(x):
    return (x + 1) * (x - 2) * (x + 3) * (x - 4) * (x + 5) * (x - 6)

def G(x):
    return x**6 - 7 * x**3 + 25

# Single test between F(x) and G(x)
def single_step(I):
    r = random.choice(I)  # Pick a random point
    return F(r) == G(r)  # Return True if F(r) == G(r), else False

# Bayes' theorem to update belief
def update_belief(prior, p_error):
    """
    Update the belief P(C|Y) using Bayes' theorem.
    """
    # P(Y|C) = 1 (always correct if polynomials are equal)
    p_y_given_c = 1
    
    # Apply Bayes' theorem
    posterior = (p_y_given_c * prior) / (p_y_given_c * prior + p_error * (1 - prior))
    return posterior

# Iterative Bayes process
def iterative_bayes(I, k, p_error):
    prior = 0.5  
    print(f"Initial prior: {prior}")
    
    for i in range(k):
        test_result = single_step(I) 
        if test_result:  
            prior = update_belief(prior, p_error) 
            print(f"Posterior after test {i+1}: {prior:.4f}")
        else:
            print(f"Test {i+1}: F(x) and G(x) are NOT equivalent")
            return "F(x) and G(x) are NOT equivalent"  
    
    return prior  

# Example usage
I = list(range(1, 51)) 
k = 5  
p_error = 1 / 100 

final_posterior = iterative_bayes(I, k, p_error)

