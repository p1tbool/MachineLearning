import numpy as np
import gc
inputs = np.array([
    [1,1,1],
    [1,0,1],
    [0,1,1],
    [0,0,1]])


results = np.array([[0],[1],[1],[0]])

def createWeigth(elements, neuronsInHiddenLayers):
    return [
        np.random.random((len(elements[0]),neuronsInHiddenLayers)),
        np.random.random((neuronsInHiddenLayers,1))
    ]
    
def sigmoid(x,derivative=False):
    if (derivative):
        return x * (1-x)
    return 1 /(1+ np.exp(-x))

def tanh(x, derivative=False):
    if (derivative):
        return x* (1-x)
    return (2/(1+np.exp((-2)*x)))-1

def feedforward(inputValues, activationFunction, weight1, weight2):
    w_sum1 = np.dot(inputValues, weight1)
    res_1 = activationFunction(w_sum1)

    w_sum2 = np.dot(res_1, weight2)
    res_2  = activationFunction(w_sum2)

    return (res_1, res_2)


def brainPower(inputs, wl0l1, wl1l2, learning_rate, activationFunction, iteration):
    for _ in range(iteration):
        res1, res2 = feedforward(inputs, activationFunction, wl0l1, wl1l2)
        delta_error = res2-results
        error = ((1/2)*(np.power((delta_error),2))) #??
        
        #backfeeding the error
        ## output to hidden
        delta_sum = activationFunction(res2, True)
        delta__weigth = res1
        delta_output_layer = np.dot(delta__weigth.T,(delta_error*delta_sum))
        
        #hidden to inputs
        delta_act_hidden = np.dot(delta_error * delta_sum, wl1l2.T)
        delta_sum_hidden = activationFunction(res1, True)
        delta_weight_Layer1 = inputs
        delta_hidden_layer  = np.dot(delta_weight_Layer1.T, delta_act_hidden*delta_sum_hidden)
        
        wl0l1 = wl0l1 - learning_rate * delta_hidden_layer
        wl1l2 = wl1l2 - learning_rate * delta_output_layer
        
        res1, res2 = feedforward(inputs, activationFunction, wl0l1, wl1l2)
        
        delta_error = res2-results
        error = ((1/2)*(np.power((delta_error),2))) #??
        
    return (sum(error),wl0l1)

def brainPowerMomentum(inputs, wl0l1, wl1l2, learning_rate, activationFunction, iteration, momentum):
    previousDeltaHL, previousDeltaOL = 0, 0
    
    for _ in range(iteration):
        res1, res2 = feedforward(inputs, activationFunction, wl0l1, wl1l2)
        delta_error = res2-results
        
        error = ((1/2)*(np.power((delta_error),2))) #?? never used
        
        #backfeeding the error
        ## output to hidden
        delta_sum = activationFunction(res2, True)
        delta__weigth = res1
        delta_output_layer = np.dot(delta__weigth.T,(delta_error*delta_sum))
        
        #hidden to inputs
        delta_act_hidden = np.dot(delta_error * delta_sum, wl1l2.T)
        delta_sum_hidden = activationFunction(res1, True)
        delta_weight_Layer1 = inputs
        delta_hidden_layer  = np.dot(delta_weight_Layer1.T, delta_act_hidden*delta_sum_hidden)
        
        previousDeltaHL = (learning_rate * delta_hidden_layer + previousDeltaHL * momentum)
        previousDeltaOL = (learning_rate * delta_output_layer + previousDeltaOL * momentum)
        
        wl0l1 = wl0l1 - previousDeltaHL
        wl1l2 = wl1l2 - previousDeltaOL
                
        res1, res2 = feedforward(inputs, activationFunction, wl0l1, wl1l2)
        
        delta_error = res2-results
        error = ((1/2)*(np.power((delta_error),2))) #??
        
    return (sum(error),wl0l1)
    
    

def test(learning_rate, iteration, hiddenLayerSize, inputs):

    
    witoh, whtoout = createWeigth(inputs, hiddenLayerSize)
    
    error, w = brainPower(inputs, witoh, whtoout, learning_rate,sigmoid, iteration)
    message = "BKFd Step {} loop {} #hn {} func {} error {:.5f}"
    print(message.format(learning_rate, iteration, hiddenLayerSize, "Sigmoid",error[0]))
    
    error, w = brainPower(inputs, witoh, whtoout, learning_rate,tanh, iteration)
    print(message.format(learning_rate, iteration, hiddenLayerSize, "tanh",error[0]))
    
    message = "Mom  Step {} loop {} #hn {} func {} error {:.5f}"
    error, w = brainPowerMomentum(inputs, witoh, whtoout, learning_rate,sigmoid, iteration,0.1)
    print(message.format(learning_rate, iteration, hiddenLayerSize, "sigmoid",error[0]))
    
    error, w = brainPowerMomentum(inputs, witoh, whtoout, learning_rate,tanh, iteration,0.1)
    print(message.format(learning_rate, iteration, hiddenLayerSize, "tanh",error[0]))
    
test(1,500,5,inputs)
test(2,500,5,inputs)
test(3,500,5,inputs)
test(5,500,5,inputs)
test(8,500,5,inputs)

test(3,300,2,inputs)
test(3,300,3,inputs)
test(3,300,4,inputs)
test(3,300,5,inputs)
test(3,300,6,inputs)
test(3,300,7,inputs)

gc.collect()