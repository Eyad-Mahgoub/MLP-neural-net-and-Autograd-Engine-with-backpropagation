from NN import MLP

xs = [                      # Dataset
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0] # Desired Target

# ---------------------------------------
# Initialise MLP
# 
#
# Input Layer: 3 neurons for 3 inputs
# Hidden layer: 2 layers of 4 neurons each
# Output Layer: 1 Neuron
#
# ---------------------------------------
n = MLP(3, [4, 4, 1]) 


# ---------------------------------------
#           Model Optimisation
# ---------------------------------------
for k in range(200):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt ,yout in zip(ys, ypred)])

    # reset grads
    for p in n.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()
    
    # update
    for p in n.parameters():
        p.data += -0.01 * p.grad
    
    print(k, loss.data)