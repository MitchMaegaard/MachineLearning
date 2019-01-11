
# coding: utf-8

# In[19]:


#%matplotlib inline

# for statisitcal methods
import math
import pandas as pd
import numpy as np
import os
import random

import warnings # for cleaner output

# for plotting -- keep out when running in ternminal
#import matplotlib.pyplot as plt
#import seaborn as sns; sns.set(font_scale=1.2, style='darkgrid')


# In[2]:


def DecisionTree(verbose=False):
    # takes all input from the user command line: data (pre-specified path)
    
    out = '' # initialize message output string
    
    train_max = 1000
    
    # assert: train_size must be positive multiple of 250 that is <= 1000
    size_mssg = '\nPlease enter a training set size (a positive multiple of 250 that is <= 1,000): '
    train_size = int(input(size_mssg))
    if (train_size % 250 != 0):
        raise ValueError('Training size must be a positive int and a multiple of 250.')
    if(train_size > 1000):
        raise ValueError('Training size must be less than 1,000.')
    
    out += size_mssg + str(train_size)
    
    # assert: train_step must be {10,25,50}
    step_mssg = '\nPlease enter a training increment (either 10, 25, or 50): '
    train_step = int(input(step_mssg))
    if train_step not in set([10,25,50]):
        raise ValueError('Training step size must be either 10, 25, or 50.')
    
    out += step_mssg + str(train_step) + '\n'
    
    # load property information -- not necessary for implementation
    #print('\nLoading Property Information from file.')
    # load data
    out += '\nLoading Data from database.\n'
    data = pd.read_csv(os.path.join(os.path.abspath('.'),
                                    'hw01', 'input_files', 'mushroom_data.txt'),
                       sep=' ', header=None)
    
    # getting warnings for appending items to data with '.' -- leave out for now
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # set up parameters that other functions will use
    data.goal = data.shape[1]-1 # should return the last column
    data.attrs = list(range(len(data.values[0]))) # 'attributes' aka column numbers
    data.attrnames = list(range(len(data.values[0]))) # names of columns -- for our case, just numbers
    data.examples = data.values # data.examples[0] should return the first row
    data.inputs = [a for a in data.attrs if a != data.goal] # should return attributes w/out the goal
    data.vals = list(map(lambda x: list(set(x)), zip(*data.examples))) # hold all values per specified column
    
    # specify training set
    out += '\nCollecting set of ' + str(train_size) + ' training examples.\n'
    
    # set aside values for training and testing
    # must do this before looping so we can retain consitent accuracy measures
    data_train = data.sample(n=train_size, random_state=3297)
    data_test = data.drop(data_train.index)
    
    # iterate until all examples in training set are used:
    train_sub = train_step
    # store training set size and accuracy
    size_vals = []
    acc_vals = []
    while (train_sub <= train_size) and (train_sub <= train_max):
        # run model with train_step from training set
        out += '\nRunning with ' + str(train_sub) + ' examples in training set.\n'
        
        data_train_sub = data_train.sample(n=train_sub, random_state=3297)
        
        train_model = decision_tree_learning(data, data_train_sub.values, data.inputs)
        
        test_model = test(train_model, data, examples=data_test.values)
        
        accuracy = round(100*(test_model/len(data_test)),4)
        
        # collect num correct classifications (success rate)
        out += '\nGiven current tree, there are ' + str(int(test_model))         + ' correct classifications\n\tout of ' + str(len(data_test))         + ' possible (success rate of ' + str(accuracy) + ' percent).\n'
        
        size_vals.append(train_sub)
        acc_vals.append(accuracy)
        
        train_sub += train_step
    
    # only include tree print-out if verbose is specified
    if verbose==True:
        out += '\n\t------------------'         + '\n\tFinal Decision Tree'         + '\n\t------------------\n'         + train_model.display()
    
    out += '\n\t----------'     + '\n\tStatistics'     + '\n\t----------\n\n'
    
    # print out size and accuracy values
    for i in range(len(size_vals)):
        out += 'Training set size: ' + str(size_vals[i])         + '. Success: ' + str(acc_vals[i]) + ' percent.\n'
    
    return out, size_vals, acc_vals


# In[3]:


def decision_tree_learning(data, examples, attributes, parent_examples=()):
    
    # if no examples are left, there are no observations w/this combo of attributes
    # return default probability from 'plurality_value()'
    if len(examples) == 0:
        return plurality_value(data, parent_examples)
    
    # base case for all remaining examples being of the same class (no further iterations)
    elif same_classification(data, examples):
        return DecisionLeaf(examples[0][data.goal]) # make classification
    
    # worst case scenario -- checking for same discription but different classification
    # best we can do here is return prob. from 'plurality_value()'
    elif len(attributes) == 0:
        return plurality_value(data, examples)
    
    # if some pos. and neg. eamples left, choose best attribute to split them
    else:
        best = importance(data, attributes, examples)
        tree = DecisionSplit(best, data.attrnames[best], plurality_value(data, examples))
        
        for(v, examples_i) in split_by(data, best, examples):
            subset = [a for a in attributes if a != best] # filter out the best attribute
            subtree = decision_tree_learning(data, examples_i, subset, examples) # build subtree on remaining attrs
            tree.add(v, subtree)
        return tree


# In[4]:


def importance(data, attributes, examples):
    # choose attribute with highest entropy (information gain)
    
    fn = lambda a: - information_gain(data, a, examples)
    
    most_gain = fn(attributes[0]) # initialize highest entropy to what we gain from our first example
    n = 0
    
    for a in attributes:
        a_gain = fn(a)
        if a_gain < most_gain:
            most_gain = a_gain
            high = a
            n = 1
        elif a_gain == most_gain:
            n += 1
            if random.randrange(n) == 0:
                    high = a
    return high


# In[5]:


def plurality_value(data, examples):
    # select most common output value among a set of examples
    # take the majority (for binary) -- 'plurality' for multi-class

    #seq = data.vals[data.goal]
    fn = lambda v: - count(data.goal, v, examples)
    
    high_vote = fn(data.vals[data.goal][0]) # keep track
    n = 0
    
    for c in data.vals[data.goal]:
        new_vote = fn(c)
        if new_vote < high_vote:
            high_vote = new_vote
            plurality = c
            n = 1
        elif new_vote == high_vote:
            n += 1
            if random.randrange(n) == 0:
                    plurality = c
                    
    return DecisionLeaf(plurality)


# In[6]:


def information_gain(data, attributes, examples):
    
    N = float(len(examples)) # same as p+n (count all examples)
    
    # calculate remainder: R(A) = sum((pk+nk)/(p+n))*B(pk/(pk+nk))
    remainder = 0
    for (v, examples_i) in split_by(data, attributes, examples):
        remainder += (len(examples_i) / N) * entropy(data, examples_i)
    
    # calculate gain: G(A) = B(p/(p+k)) - R(A)
    return entropy(data, examples) - remainder


# In[7]:


def entropy(data, examples):
    # keep a count of our classifications
    vals = [count(data.goal, v, examples) for v in data.vals[data.goal]]
    # remove 0's (if any)
    vals = [v for v in vals if v!= 0]
    # normalize -- num bits as representation
    s = float(sum(vals))
    if s != 1.0: vals = [v/s for v in vals] # create probabilities -- p(x) -- by dividing by totals
    return sum([-v*math.log2(v) for v in vals]) # H(x) = -sum(p(x)*log_2(p(x)))


# In[8]:


def split_by(data, attributes, examples):
    # split data based on a specific attribute
    return [(v, [e for e in examples if e[attributes] == v])
           for v in data.vals[attributes]]


# In[9]:


def count(attribute, value, examples):
    # count num examples with examples[attribute] = value
    return sum(e[attribute] == value for e in examples)


# In[10]:


def same_classification(data, examples):
    # store the classification of our first example
    firstclass = examples[0][data.goal]
    # check remaining examples to see if they're the same
    return all(e[data.goal] == firstclass for e in examples)


# In[11]:


def test(predict, data, examples=None):
    # check our classifications on the remaining data
    # 'predict' is our fitted model
    if examples is None: examples = data.examples
    
    if len(examples) == 0: return 0
    
    correct = 0 # keep track of how many we classify correctly
    for example in examples:
        # supervised learning -- make use of the classification that we KNOW
        expected = example[data.goal]
        # retrieve values from our model
        output = predict([attr_i if i in data.inputs else None
                          for i, attr_i in enumerate(example)])
        # compare to what we THINK
        if output == expected: correct += 1

    return correct


# In[12]:


class DecisionSplit:
    
    def __init__(self, attributes, attrname=None, default_child=None, branches=None):
        # constructor -- specify what attribute this node tests
        self.attributes = attributes
        self.attrname = attrname or attributes
        self.default_child = default_child
        self.branches = branches or {}
        
    def __call__(self, example):
        # classify an example using the attribute and branches
        attribute_val = example[self.attributes]
        if attribute_val in self.branches:
            return self.branches[attribute_val](example)
        else:
            # if attribute is unknown, return the default class
            return self.default_child(example)
    
    def add(self, value, subtree):
        # add a branch -- if attributes = value, go to given subtree
        self.branches[value] = subtree
    
    def display(self, indent=0):
        # attempt to build tree with 'verbose' mode -- working for print-out, not so hot for string
        name = self.attrname
        branch = ''
        branch += 'Attr: ' + str(name)
        for(val, subtree) in self.branches.items():
            branch += ' '*4*indent + str(name) + ': ' + str(val) + '~~>'
            branch += str(subtree.display(indent + 1))
        branch += '\n'
    
    def __repr__(self):
        return('DecisionSplit({0!r}, {1!r}, {2!r})'
               .format(self.attributes, self.attrname, self.branches))


# In[13]:


class DecisionLeaf:
    # leaf of the tree holds the results
    
    def __init__(self, result):
        self.result = result
    
    def __call__(self, example):
        return self.result
    
    def display(self, indent=0):
        if self.result == 'p':
            leaf = 'Class: Poison'
        elif self.result == 'e':
            leaf = 'Class: Edible'
        return leaf
    
    def __repr__(self):
        return repr(self.result)


# In[24]:


# save results from 2 trial runs
tree1, sz1, per1 = DecisionTree() # run first example with S=250 and I=10
print(tree1)
# only include for saving results
'''
file1 = open('output_run01','w') # specify 'w' for writing to the file, truncating first
file1.write(tree1)
file1.close()
'''

# plot training accuracy results
'''
plt.plot(sz1, per1)
plt.xlabel('\nSize of Training Set')
plt.ylabel('Percent Correct')
plt.show()
#plt.savefig('graph_run01')
'''


# In[16]:


tree2, sz2, per2 = DecisionTree() # second example with S=1,000 and I=50
print(tree2)
'''
file2 = open('output_run02','w')
file2.write(tree2)
file2.close()
'''

'''
plt.plot(sz2, per2)
plt.xlabel('\nSize of Training Set')
plt.ylabel('Percent Correct')
plt.show()
plt.savefig('graph_run02')
'''

