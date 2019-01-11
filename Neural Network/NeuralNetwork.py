
# coding: utf-8

# In[1]:


import os # help set file paths
import math # help with calculations
import random


# In[2]:


def sigmoid(x):
    # sigmoid activation
    return 1 / (1 + math.exp(-x))

def sigmoid_deriv(x):
    # sigmoid derivative
    # input: x = node.value -- sigmoid activation applied
    return x*(1-x)

def dotprod(x, y):
    # dot product of 2 lists
    return sum([i*j for (i, j) in zip(x, y)])


# In[3]:


class Neuron:
    # represents a single neuron, holding:
        # inputs = incoming connections
        # weights = weights of inputs
    def __init__(self, activation=sigmoid, inputs=None, weights=None):
        self.inputs = inputs or []
        self.weights = weights or []
        self.value = None
        self.activation = activation


# In[4]:


def save_network(file_path, network, test_file, accuracy):
    
    f = open(file_path, 'w')
        
    # save file name, test file, and accuracy
    save_name = file_path[file_path.rfind('/')+1:]
    nn_specs = str(save_name) + '\n' + str(test_file) + '\n' + str(accuracy) + '\n'
    
    # save layer sizes
    for layer in range(len(network)):
        sizes = len(network[layer])
        #print('\nSizes: {}'.format(sizes))
        nn_specs += str(sizes) + ','
    nn_specs = nn_specs[:-1] # take off the extra comma
    
    for layer in network[1:]:
        #print('\n\tLayer')
        nn_specs += '\n'
        for node in layer:
            #print('\nWeights: {}'.format(node.weights))
            weight = node.weights
            nn_specs += str(weight) + ','
        nn_specs = nn_specs[:-1]
    
    f.write(nn_specs)
    f.close()


# In[8]:


def load_network(filename):
    with open(filename) as f:
        contents = f.read().splitlines()
        f.close()
    
    # pull data line by line
    name = contents[0]
    test_file = contents[1]
    accuracy = contents[2]
    topology = list(map(int, contents[3].split(',')))
    # remaining data represents the weights
    weights = []
    for i in range(4, len(contents)):
        weights.append(contents[i])
    
    layer_weights = []
    for i in range(len(weights)):
        tmp = weights[i].split('],')
        arr = []
        for j in range(len(tmp)):
            w = tmp[j][1:].replace(']','').split(',')
            if w != ['']:
                arr.append(list(map(float, w)))
        layer_weights.append(arr)
    
    layer_sizes = [[num_units] for num_units in topology]
    network = rebuild(layer_sizes, layer_weights)
    
    return network, topology, test_file, accuracy


# In[9]:


def rebuild(layer_sizes, weights):
    
    # set up neurons in each layer with inputs and weights
    NN = [[Neuron(sigmoid) for size in range(layer[0])] for layer in layer_sizes]
    
    # connect hidden layers
    for layer in range(1, len(NN)-1):
        for node in NN[layer][:-1]: # for each node in layer up to the last one
            for in_layer in NN[layer-1]: # for each node in the previous layer
                node.inputs.append(in_layer) # tack on the incoming values
                node.weights.append([weights[layer-1][i] for i in range(len(weights[layer-1]))])
        
    # connect last hidden layer to output layer
    for node in NN[-1]:
        for in_layer in NN[-2]:
            node.inputs.append(in_layer)
            node.weights.append([weights[-1][i] for i in range(len(weights[-1]))])
    
    return NN


# In[10]:


def data_scrub(file, res):
    # load data -- last res^2 values are pixels (0, 255), first several are histograms (0, 1)
    with open(file) as infile:
        data = infile.read().splitlines()[5:] # skip first 5 lines -- info about dataset
    
    # split data into pixels and output
    pixels = []
    labels = []
    for ex in range(len(data)):
        temp = data[ex].split(') (')
        pixels.append(temp[0].replace('(','').split(' '))
        labels.append(temp[1].replace(')','').split(' '))
    
    pixels = [[float(j) for j in i] for i in pixels]
    labels = [[float(j) for j in i] for i in labels]
    
    # normalize pixels
    for i in range(len(pixels)):
        pixels[i][-res**2:] = [x / 255. for x in pixels[i][-res**2:]]
        
    # combine pixels and labels into a single list
    examples = list(map(list, zip(pixels, labels)))
    
    return examples # pixels, labels


# In[11]:


def BackPropLearning(examples, network):
    # inputs: examples -- set of examples, each with input x and output y
            # network -- multilayer network with:
                    # - L layers
                    # - weights wij
                    # - activation function g
    # local vars: delta -- a vector of errors
                # indexed by network node
    
    input_layer = network[0] # first layer
    output_layer = network[-1] # last layer
    out_size = len(output_layer) # num. output neurons -- 14 for our examples
    num_layers = len(network)
    
    split_exs = list(zip(*examples))
    X = split_exs[0]
    [X[i].append(1.0) for i in range(len(X))] # add bias column
    Y = split_exs[1]
    
    # initialize weights (done in network function)
    
    epochs = 1000
    # repeat until: no. iterations reaches 1,000 OR max error on a pass through data < 0.01
    for epoch in range(epochs):
        #print('Epoch {}'.format(epoch))
        max_error = [] # keep track of error terms for all epoch
        
        # iterate over each example (x,y)
        for ex in range(len(examples)):
            x = X[ex] # grab input pixel vector
            y = Y[ex] # grab output vector
            
            # initialize input neuron values -- (0,1)
            for x_i, node in zip(x, input_layer): # for each node in input
                node.value = x_i # a_i = x_i
            
            # Propagate the inputs forward to compute outputs
            for layer in network[1:-1]: # for l=2 to L
                bias = layer[-1] # account for bias node
                for node in layer[:-1]: # for each node in layer l, but NOT the bias!
                    a_i = [a.value for a in node.inputs]
                    in_j = dotprod(node.weights, a_i) # in_j = sum(w_ij*a_i)
                    node.value = node.activation(in_j) # a_j = g(in_j)
                bias.value = 1.0
            
            # activate output layer separately -- no bias term
            for node in output_layer:
                a_i = [a.value for a in node.inputs]
                in_j = dotprod(node.weights, a_i)
                node.value = node.activation(in_j)
            
            # initialize delta (error vector)
            delta = [[] for _ in range(num_layers)]
            
            # sum(y_i - a_i)
            error = [(y[i] - output_layer[i].value) for i in range(out_size)]
            
            # Propagate deltas backward from output layer to input layer
            
            # for each node j in the output layer do:
                # delta[j] = g'(in_j)*(y_j-a_j)
            delta[-1] = [sigmoid_deriv(output_layer[j].value) * error[j] for j in range(out_size)]
            
            # for l=L-1 to 1 do:
            num_hidden = num_layers - 2 # if perceptron, = 0
            
            # perceptron/base calculation -- connect to hidden node
            hidden = network[num_hidden]
            nx_layer = network[num_hidden+1]
            w = [[node.weights[i] for node in nx_layer] for i in range(len(hidden))]
            delta[num_hidden] = [sigmoid_deriv(hidden[j].value) * dotprod(w[j], delta[num_hidden+1]) for j in range(len(hidden))]
            
            # perceptron will NOT calculate anything here
            # leave hidden node out of weight update
            for l in range(num_hidden-1, 0, -1): # -1 to go backwards through layers
                hidden_layer = network[l] # current hidden layer
                next_layer = network[l+1] # hidden layer 1 place before
                # for each node i in layer l do:
                    # delta[i] = g'(in_i)*sum(w_ij*delta[j])
                w = [[node.weights[i] for node in next_layer[:-1]] for i in range(len(hidden_layer))]
                delta[l] = [sigmoid_deriv(hidden_layer[j].value) * dotprod(w[j], delta[l+1]) for j in range(len(hidden_layer))]
            
            # Update every weight in network using deltas
            # for each weight w_ij in network do:
            for l in range(1, num_layers-1):
                curr_layer = network[l]
                a = [node.value for node in network[l-1]]
                for j in range(len(curr_layer)-1):
                    # w_ij = w_ij + alpha*a_i*delta[j]
                    update = [delta[l][j] * a_i for a_i in a]
                    curr_layer[j].weights = [sum(w) for w in zip(curr_layer[j].weights, update)]
            
            last_layer = network[-1]
            a = [node.value for node in network[-2]]
            for i in range(len(last_layer)):
                update = [delta[-1][i] * a_i for a_i in a]
                last_layer[i].weights = [sum(z) for z in zip(last_layer[i].weights, update)]
            
            max_error.append(max(error))
        
        # early stop if we minimize error
        if max(max_error) < 0.1:
            break
    
    return network


# In[12]:


def network(input_units, hidden_layer_sizes, output_units):
    
    # perceptron if hidden_layer_sizes = []
    layer_sizes = [input_units] + hidden_layer_sizes + [output_units]
    
    # set up neurons in each layer with inputs and weights
    #NN = [[Neuron(sigmoid) for size in range(layer)] for layer in layer_sizes]
    NN = [[Neuron(sigmoid) for node in range(layer+1)] for layer in layer_sizes[:-1]]
    NN.append([Neuron(sigmoid) for node in range(layer_sizes[-1])])
    
    # connect hidden layers
    for layer in range(1, len(NN)-1):
        for node in NN[layer][:-1]: # for each node in layer up to the last one
            for in_layer in NN[layer-1]: # for each node in the previous layer
                node.inputs.append(in_layer) # tack on the incoming values
                node.weights.append(random.random()) # set random weights
    
    # connect last hidden layer to output layer
    for node in NN[-1]:
        for in_layer in NN[-2]:
            node.inputs.append(in_layer)
            node.weights.append(random.random()) # or try random.uniform(0.0, 0.001)
    
    return NN


# In[13]:


def test(examples, network):
    
    input_layer = network[0]
    
    # split data like training set
    split_exs = list(zip(*examples))
    X = split_exs[0]
    [X[i].append(1.0) for i in range(len(X))] # add bias column
    Y = split_exs[1]
    
    # initialize lists to store predicted and actual values
    preds, actual = [], []
    correct = 0 # keep track of how many we correctly identify
    
    for ex in range(len(examples)):
        x = X[ex] # input
        y = Y[ex] # goal
        
        # activate input layer
        for x_i, node in zip(x, input_layer):
            node.value = x_i

        for layer in network[1:-1]: # for l=2 to L
            bias = layer[-1] # account for bias node
            for node in layer[:-1]: # for each node in layer l, but NOT the bias!
                a_i = [a.value for a in node.inputs]
                in_j = dotprod(node.weights, a_i) # in_j = sum(w_ij*a_i)
                node.value = node.activation(in_j) # a_j = g(in_j)
            bias.value = 1.0
            
        # activate output layer separately -- no bias term
        for node in network[-1]:
            a_i = [a.value for a in node.inputs]
            in_j = dotprod(node.weights, a_i)
            node.value = node.activation(in_j)
        
        output_layer = network[-1]
        vals = [output_layer[i].value for i in range(len(output_layer))]
        
        # find index of predicted and store in list
        preds.append(vals.index(max(vals)))
        #preds.append(output_layer.index(max(output_layer, key=lambda node: node.value)))
        
        # find index of actual and store in list
        actual.append(y.index(max(y)))
        # check if we have a match at this element
        if preds[ex] == actual[ex]:
            correct += 1
    
    score = round(100 * (correct / len(examples)), 1) # accuracy in percentage to 1 decimal
    
    return score


# In[16]:


def NeuralNet():
    
    # set data path
    path = os.path.join(os.path.abspath('.'),'hw03')
    
    while True:
        
        # prompt user for what they want to do
        action = input('Enter L to load trained network, T to train a new one, Q to quit: ').upper()
        
        # if user wants to train a new NN
        if action == 'T': # handle upper and lower cases
            
            # prompt user for data resolution {5, 10, 15, 20}
            resolution = input('Resolution of data (5/10/15/20): ')
            if resolution not in ['5','10','15','20']:
                raise ValueError('Image resolution must be in the set {5, 10, 15, 20}.')
            if resolution == '5':
                resolution = '05'
            
            # prompt user for number of hidden layers
            num_hidden = int(input('Number of hidden layers: '))
            if num_hidden not in range(11):
                raise ValueError('Number of hidden layers must be an integer between 0 and 10.')
            
            # get size for each hidden layer
            hidden_layer_sizes = []
            # if hidden is 0, layer_sizes will just be an empty list
            # and user will not be prompted for a layer size
            for layer in range(num_hidden):
                hidden_layer_sizes.append(int(input('Size of hidden layer ' + str(layer+1) + ': ')))
            
            print('\nInitializing network...')
            
            train_file = 'trainSet_' + resolution + '.dat'
            print('Training on {}...'.format(train_file))
            train_examples = data_scrub(os.path.join(path,'trainSet_data',train_file), int(resolution))
            
            # run model
            in_units = len(train_examples[0][0]) # 65 for 5x5 (assume length will be = for all examples)
            out_units = len(train_examples[0][1]) # 14 for ALL examples
            init_net = network(in_units, hidden_layer_sizes, out_units)
            learned_net = BackPropLearning(train_examples, init_net)
            
            # test model
            test_file = 'testSet_' + resolution + '.dat'
            print('Testing on {}...\n'.format(test_file))
            test_examples = data_scrub(os.path.join(path,'testSet_data',test_file), int(resolution))
            score = test(test_examples, learned_net)
            print('Accuracy Achieved: {}%\n'.format(score))
            
            # check if user wants to save the model
            save_check = input('Save network (Y/N)? ').upper()
            if save_check == 'Y':
                file_save = input('File-name: ')
                print('Saving network...')
                # save file to path
                save_network(os.path.join(path, file_save), learned_net, test_file, score)
                print('Network saved to file:', file_save)
                continue
            elif save_check == 'N':
                continue
            # handle case where user made an error
            else:
                raise ValueError('Options to save are either Y or N.')
            
        # if user wants to load an existing NN
        elif action == 'L':
            # prompt user for neural network file name
            nnfile = input('Network file-name: ')
            # read file
            pretrainedNN, topology, tested_file, trained_acc = load_network(os.path.join(path, nnfile))
            print('\nLoading network from {}...'.format(nnfile))
            print('Input layer size: {} '.format(topology[0]-1)) # -1 for bias
            hid = ''
            for i in range(len(topology[1:-1])):
                hid += str(topology[1:-1][i]-1) + ', '
            print('Hidden layer sizes: {} '.format(hid[:-2])) # remove last comma
            print('Output layer size: {}'.format(topology[-1]))
            
            print('Testing on {}...'.format(tested_file))
            test_size = tested_file.split('_')[1].split('.')[0]
            #score = test(data_scrub(os.path.join(path,'testSet_data',tested_file), int(test_size)), pretrainedNN)
            print('Accuracy achieved: {}%\n'.format(trained_acc))
            
            continue # keep looping
        
        # if user wants to stop having fun
        elif action == 'Q':
            break # exit while loop
        
        else:
            print('Please enter a valid command.')
            continue

    print('\nGoodbye.')


# In[15]:


myNN = NeuralNet()


# In[ ]:


def main():
    NN_model = NeuralNet()


# In[ ]:


if __name__ == 'main':
    main()

