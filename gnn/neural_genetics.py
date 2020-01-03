import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


# Could make a generalized instance of the Model Keras class for a GNN
# New Type of Neural Network
class GeneticNeuralNetwork(Sequential):
    # Constructor
    def __init__(self, child_weights=None):
        # Initialize Sequential Model Super Class
        super().__init__()
        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are created and randomly generated
            layer1 = Dense(4, input_shape=(4,), activation='sigmoid')
            layer2 = Dense(2, activation='sigmoid')
            layer3 = Dense(1, activation='sigmoid')
            # Layers are added to the model
            self.add(layer1)
            self.add(layer2)
            self.add(layer3)
        # If weights are provided set them within the layers
        else:
            # Set weights within the layers
            self.add(
                Dense(
                    4,
                    input_shape=(4,),
                    activation='sigmoid',
                    weights=[child_weights[0], np.zeros(4)])
                )
            self.add(
                Dense(
                 2,
                 activation='sigmoid',
                 weights=[child_weights[1], np.zeros(2)])
            )
            self.add(
                Dense(
                 1,
                 activation='sigmoid',
                 weights=[child_weights[2], np.zeros(1)])
            )

    # Function for forward propagating a row vector of a matrix
    def forward_propagation(self, X_train, y_train):
        # Forward propagation
        y_hat = self.predict(X_train.values)
        # Compute fitness score
        self.fitness = accuracy_score(y_train, y_hat.round())

    # Standard Backpropagation
    def compile_train(self, epochs):
        self.compile(
                      optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy']
                      )
        self.fit(X_train.values, y_train.values, epochs=epochs)


# Chance to mutate weights
def mutation(child_weights):
    # Add a chance for random mutation
    selection = random.randint(0, len(child_weights)-1)
    mut = random.uniform(0, 1)
    if mut >= .5:
        child_weights[selection] *= random.randint(2, 5)
    else:
        # No mutation
        pass


# Crossover traits between two Genetic Neural Networks
def dynamic_crossover(nn1, nn2):
    # Lists for respective weights
    nn1_weights = []
    nn2_weights = []
    child_weights = []
    # Get all weights from all layers in the first network
    for layer in nn1.layers:
        nn1_weights.append(layer.get_weights()[0])

    # Get all weights from all layers in the second network
    for layer in nn2.layers:
        nn2_weights.append(layer.get_weights()[0])

    # Iterate through all weights from all layers for crossover
    for i in range(0, len(nn1_weights)):
        # Get single point to split the matrix in parents based on # of cols
        split = random.randint(0, np.shape(nn1_weights[i])[1]-1)
        # Iterate through after a single point and set the remaing cols to nn_2
        for j in range(split, np.shape(nn1_weights[i])[1]-1):
            nn1_weights[i][:, j] = nn2_weights[i][:, j]

        # After crossover add weights to child
        child_weights.append(nn1_weights[i])

    # Add a chance for mutation
    mutation(child_weights)

    # Create and return child object
    child = GeneticNeuralNetwork(child_weights)
    return child


# Read Data
data = pd.read_csv('assets/banknote.csv')
# Create Matrix of Independent Variables
X = data.drop(['Y'], axis=1)
# Create Vector of Dependent Variable
y = data['Y']
# Create a Train Test Split for Genetic Optimization
X_train, X_test, y_train, y_test = train_test_split(X, y)
# Create a List of all active GeneticNeuralNetworks
networks = []
pool = []
# Track Generations
generation = 0
# Initial Population
n = 20

# Generate n randomly weighted neural networks
for i in range(0, n):
    networks.append(GeneticNeuralNetwork())

# Cache Max Fitness
max_fitness = 0

# Max Fitness Weights
optimal_weights = []

# Evolution Loop
while max_fitness < .9:
    # Log the current generation
    generation += 1
    print('Generation: ', generation)

    # Forward propagate the neural networks to compute a fitness score
    for nn in networks:
        # Propagate to calculate fitness score
        nn.forward_propagation(X_train, y_train)
        # Add to pool after calculating fitness
        pool.append(nn)

    # Clear for propagation of next children
    networks.clear()

    # Sort based on fitness
    pool = sorted(pool, key=lambda x: x.fitness)
    pool.reverse()

    # Find Max Fitness and Log Associated Weights
    for i in range(0, len(pool)):
        # If there is a new max fitness among the population
        if pool[i].fitness > max_fitness:
            max_fitness = pool[i].fitness
            print('Max Fitness: ', max_fitness)
            # Reset optimal_weights
            optimal_weights = []
            # Iterate through layers, get weights, and append to optimal
            for layer in pool[i].layers:
                optimal_weights.append(layer.get_weights()[0])
            print(optimal_weights)

    # Crossover, top 5 randomly select 2 partners for child
    for i in range(0, 5):
        for j in range(0, 2):
            # Create a child and add to networks
            temp = dynamic_crossover(pool[i], random.choice(pool))
            # Add to networks to calculate fitness score next iteration
            networks.append(temp)

# Create a Genetic Neural Network with optimal initial weights
gnn = GeneticNeuralNetwork(optimal_weights)
gnn.compile_train(10)

# Test the Genetic Neural Network Out of Sample
y_hat = gnn.predict(X_test.values)
print('Test Accuracy: %.2f' % accuracy_score(y_test, y_hat.round()))
