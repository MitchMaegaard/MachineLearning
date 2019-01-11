
# coding: utf-8

# In[1]:


import os
import pandas as pd
import sys


# In[3]:


def euclidean_dist(point1, point2):
    # use euclidean -- works well for extending to multiple dimensions
    # L^2(xi,xj) = sqrt(sum(xik-xjk)^2)
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** (.5)


# In[3]:


def nn(testdata, data):
    # base case -- take the first point in our data
    closest = data[0]
    distance = euclidean_dist(testdata, closest)
    # check other points in leaf
    for i in range(1, len(data)):
        check_close = data[i]
        check_dist = euclidean_dist(testdata, check_close)
        # set new distance, if necessary
        if check_dist < distance:
            closest = check_close
            distance = check_dist
    
    return closest, format(distance, '.6f')


# In[4]:


class Node:
    def __init__(self, data=None, box=None, left = None, right = None):
        # constructor -- specify the median value this node tests
        self.data = data
        self.box = box
        self.left = left
        self.right = right
    
    def search(self, testdata, d=0):
        col = d % testdata.shape[0] # cycle columns
        # base case to stop search
        if self.left is None and self.right is None:
            # if we have an empty leaf, we don't have a nearest neighbor!
            if not self.data:
                return str(testdata) + ' has no nearest neighbor (in an empty set).'
            else:
                close, dist = nn(testdata, self.data)
                return str(testdata) + ' is in the set ' + repr(self.data)                         + '\nNearest Neighbor: ' + str(close)                         + ' (distance = ' + str(dist) + ')'
        # search left, then right
        if testdata[col] <= self.data:
            return self.left.search(testdata, d+1)
        if testdata[col] > self.data:
            return self.right.search(testdata, d+1)
    
    def display(self):
        if self.left:
            self.left.display()
        # stop once we get to a leaf!
        if self.left is None and self.right is None:
            if self.box is None:
                print('') # don't show the leaf if it's empty
            else:
                print('Bounding Box:', repr(self.box), '\nData in leaf:', repr(self.data), '\n')
        if self.right:
            self.right.display()


# In[8]:


def build_tree(X, S, d = 0):
    # input: X = {x1,...,xn}, a set of n-dimensional data-points
    #        d = depth
    # output: tree
    # local var: S >= 1, user-defined size limit for sets
    
    n = X.shape[0] # get number of rows of data -- assume each feature has the same number of observations
    
    # calculate bounding box and add to node -- calculate here because we want to store in leaf & non-leaf nodes
    if n > 0:
        box = [(min(X[0]), min(X[1])), (max(X[0]), max(X[1]))]
    else:
        box = None
    
    # split data until each set of points has S or fewer members
    if n <= S:
        return Node(X.values.tolist(), box, None, None) # tree-node containing all elements of X
    else:
        split_feature = (d % X.shape[1]) # dimension for splitting inputs
        
        x_i = sorted(X[split_feature]) # sort feature
        mid_idx = (n - 1) // 2 # find middle value
        median = x_i[mid_idx] # get median of list
        
        X_low = X[X[split_feature] <= median] # subset of data where elements xi <= median
        X_high = X[X[split_feature] > median]
        
        N_left = build_tree(X_low, S, d + 1)
        N_right = build_tree(X_high, S, d + 1)
        return Node(median, box, N_left, N_right)


# In[9]:


def kdtree(data_file, S):
    
    S = int(S)
    
    # set path for files -- input data should be in the same location as our program
    path = os.path.join(os.path.abspath('.'),'hw02','inputData')
    
    # read in data
        # integer value d that gives dimensionality of data -- we can ignore for our implementation
        # 1+ lines consisting of d floating-point vals that define data points
    data = pd.read_csv(os.path.join(path, data_file),
                       sep=' ', skiprows = 1, header=None)
    
    # check that S is >= 1
    if S < 1:
        raise ValueError('Please choose a minimal set-size greater than or equal to 1.')
    
    # build tree
    KDTree = build_tree(round(data, 6), S)
    
    # check if user wants to visualize the tree
    visual = input('Print tree leaves? (Enter Y for yes, anything else for no): ')
    if visual == 'y':
        print('\n')
        KDTree.display() # print out KDTree leaves
        print('------------------------------\n')
    
    # check if user wants to test the tree
    test_tree = input('Test data? (Enter Y for yes, anything else to quit): ')
    if test_tree == 'y': # if yes, read in data from test file
        test_data_name = input('Name of data-file: ')
        
        test_data = pd.read_csv(os.path.join(path, test_data_name),
                               sep=' ', skiprows = 1, header=None)
        # check that the dimensionality is the same as in training
        if test_data.shape[1] != data.shape[1]:
            raise ValueError('Test data must have the same dimensions as training data!')
    
        print('\n------------------------------\n')
    
        for i in range(test_data.shape[0]):
            print(KDTree.search(test_data.values[i]), '\n')
    
    print('------------------------------\nGoodbye.')


# In[ ]:


kdtree(sys.argv[1], sys.argv[2])

