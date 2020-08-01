#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""
import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab.char2id['<pad>']
        #(echar,f,k,mword)
        echar=50
        k=5
        eword=embed_size
        f=eword
        mword=21    #max word length

        self.embeddings=nn.Embedding(len(vocab.char2id),echar,padding_idx=pad_token_idx)
        self.CNN=CNN(echar,f,k,mword)
        self.Highway=Highway(eword)
        self.Dropout=nn.Dropout(p=0.3)
        self.embed_size=embed_size
        self.mword=21


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        inmword=(input.shape)[2]
        
        #print('input',input.shape)
        input=input[:,:,(inmword-self.mword):inmword]
        #print('inputnew',input.shape)
        chembed=self.embeddings(input)
        sentence_length,batch_size,max_word_length,char_embed_size=chembed.shape
        #print('shape')
        #print(sentence_length,batch_size,max_word_length,char_embed_size)
        chembed=torch.transpose(chembed,2,3)
        chembed=chembed.reshape(sentence_length*batch_size,char_embed_size,max_word_length)
        #print('embed',chembed.shape)
        conv=self.CNN(chembed)
        #print('after conv',conv.shape)
        bat_sen,eword,_=conv.shape
        #print('conv',conv.shape)
        conv=conv.reshape(bat_sen,eword)
        high=self.Highway(conv)
        output=self.Dropout(high)
        output=output.reshape(sentence_length, batch_size, self.embed_size)
        
        #print('output',output.shape)
        return output


        ### END YOUR CODE

