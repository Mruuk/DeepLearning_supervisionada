import numpy as np

#transfer function
def stepFunction(soma):
        if (soma >= 1):
            return 1
        return 0

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

def tahnFunction(soma):
    return(np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0
        
def linearFunction(soma):
    return soma

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

test = stepFunction(2.1)
print(test)

test = sigmoidFunction(2.1)
print(test)

test = tahnFunction(2.1)
print(test)

test = reluFunction(-2.1)
print(test)

test = linearFunction(2.1)
print(test)

valores = [7.0, 2.0, 1.3]

test = softmaxFunction(valores)
print(test)