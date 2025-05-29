import random
#empirical explanation
'''
The randomized algorithm offers a much faster method to verify polynomial equality with a small chance of error.
Normally it would be computational heavy since u have to expand the polynomials.
'''
# Define the polynomial F(x) so the first part of the function
def F(x):
    return (x + 1) * (x - 2) * (x + 3) * (x - 4) * (x + 5) * (x - 6)

# Define the polynomial G(x) so the second part of the function
def G(x):
    return x**6 - 7 * x**3 + 25

# Choose a random value for r
r = random.randint(1, 100) #From a set I={1,2,…,100⋅d}, where d is the degree of the polynomials.
#The randomized algorithm evaluates 𝐹(𝑟) and G(r) at a random value of r from a set of integers

# Compute F(r) and G(r)
F_r = F(r)
G_r = G(r)

# Output the results
print(f"Random r chosen: {r}")
print(f"F({r}) = {F_r}")
print(f"G({r}) = {G_r}")

# Check if the polynomials are equal at r
if F_r == G_r:
    print(f"The polynomials F(x) and G(x) might be equal at r = {r}.")
    #We should need further check
else:
    print(f"The polynomials F(x) and G(x) are not equal at r = {r}.")
    #We can confidently say that 𝐹(𝑟) and G(r) are not equal
