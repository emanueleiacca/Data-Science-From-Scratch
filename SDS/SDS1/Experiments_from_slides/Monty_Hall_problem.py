import random

#Problem Set Up

'''
You are a contestant on a game show and presented with three doors.
Behind one door is a car (the prize), and behind the other two doors are goats.
You choose a door at random.
The host, Monty Hall, who knows what is behind the doors, opens one of the two remaining doors, revealing a goat.
Monty then gives you the option to stick with your original choice or switch to the other unopened door.
The question is: Should you stick with your original choice or switch to the other door to maximize your chances of winning the car?
'''

# Function to simulate the Monty Hall game
def monty_hall(switch):
    """
    This function simulates a single round of the Monty Hall game.
    """
    
    # Step 1: Set up the doors
    # One of the doors has a car, and the other two have goats.
    doors = [0, 0, 0]  # 0 represents a goat
    car_position = random.randint(0, 2)  # Randomly place the car behind one of the doors
    doors[car_position] = 1  # 1 represents the car

    # Step 2: Contestant makes an initial choice randomly
    contestant_choice = random.randint(0, 2)

    # Step 3: Monty opens a door with a goat
    # Monty opens one of the doors that is not the contestant's choice and doesn't have the car.
    available_doors_to_open = [i for i in range(3) if i != contestant_choice and doors[i] == 0]
    monty_opens = random.choice(available_doors_to_open)

    # Step 4: Contestant either sticks or switches
    if switch:
        # If the contestant switches, they switch to the remaining unopened door
        remaining_doors = [i for i in range(3) if i != contestant_choice and i != monty_opens]
        contestant_choice = remaining_doors[0]

    # Step 5: Check if the contestant wins
    # If the contestant's final choice is the door with the car, they win.
    return doors[contestant_choice] == 1


# Function to simulate the game multiple times
def simulate_monty_hall(num_trials, switch):
    """
    This function simulates the Monty Hall problem over multiple trials.
    """
    
    # Initialize a counter for wins
    wins = 0

    # Run the simulation for the specified number of trials
    for _ in range(num_trials):
        if monty_hall(switch):
            wins += 1

    # Calculate the probability of winning
    win_probability = wins / num_trials
    return win_probability


# Simulation parameters
num_trials = 10000  
switch_strategy = True 

# Run the simulation for both strategies 
win_probability_switch = simulate_monty_hall(num_trials, switch=True)  
win_probability_no_switch = simulate_monty_hall(num_trials, switch=False)  

print(f"Win probability when switching: {win_probability_switch * 100:.2f}%")
print(f"Win probability when not switching: {win_probability_no_switch * 100:.2f}%")
