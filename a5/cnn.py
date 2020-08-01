#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self, echar,f,k,mword):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CNN, self).__init__()
        
        
        self.conv1 = nn.Conv1d(in_channels=echar, out_channels=f, kernel_size=k, stride=1, padding=0)
        self.maxpool=nn.MaxPool1d(mword-k+1,stride=0)
        #print('mword-k+1',mword-k+1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        
        m=torch.nn.ReLU()
        

        xconv=self.conv1(x)
        xrelu=m(xconv)
        #print('xrelu',xrelu.shape)
        xpool=self.maxpool(xrelu)
        #print('xpool',xpool.shape)
 
        return xpool


### END YOUR CODE
'''

N, mword,echar  = 10, 7,4
k=5
eword=6
f=eword
#eword=256    # in ass4
#echar=50     

# Create random Tensors to hold inputs and outputs
xresh = torch.randn(N, echar,mword)
#y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = CNN(echar,f,k,mword)

y_pred=model(xresh)

#print(y_pred.shape)
#print(y_pred)
'''

