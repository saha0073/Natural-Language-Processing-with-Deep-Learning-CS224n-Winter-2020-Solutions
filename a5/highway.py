#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
### YOUR CODE HERE for part 1h
class Highway(torch.nn.Module):
    def __init__(self, D_in):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Highway, self).__init__()
        D_out=D_in
        self.proj = torch.nn.Linear(D_in, D_out)
        self.gate = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #h_relu = self.linear1(x).clamp(min=0)
        #y_pred = self.linear2(h_relu)
        xproj=self.proj(x)
        m=torch.nn.ReLU()
        xproj=m(xproj)
        
 
        xgate=self.gate(x)
        m = torch.nn.Sigmoid()
        xgate=m(xgate)
        #print('xproj',type(xproj))
        #print('xgate',type(xgate))

        xhigh=xgate*xproj+(1-xgate)*x
        

        return xhigh
### END YOUR CODE
'''
N, D_in  = 10, 3

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
#y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = Highway(D_in)

y_pred=model(x)

#print(y_pred)


 
'''
