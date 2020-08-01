#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        self.hidden_size=hidden_size
        super(CharDecoder, self).__init__()
        self.charDecoder=nn.LSTM(char_embedding_size,hidden_size)
        self.char_output_projection=nn.Linear(hidden_size,len(target_vocab.char2id))
        
        self.target_vocab=target_vocab
        padding_idx=target_vocab.char2id['<pad>']
        self.decoderCharEmb=nn.Embedding(len(target_vocab.char2id),char_embedding_size,padding_idx)
        #print('char_embedinit123',char_embedding_size)
        #self.embeddings=nn.Embedding(len(target_vocab.char2id),char_embedding_size,padding_idx)
        

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        #h0,c0=dec_hidden
    
        input_embed=self.decoderCharEmb(input)
        #print('input',type(input_embed))
        #print('len',len(self.target_vocab.char2id))
        outputs,(h_n,c_n)=self.charDecoder(input_embed,dec_hidden)
        scores=self.char_output_projection(outputs)
        
        return scores,(h_n,c_n)
        
        
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        length,batch=char_sequence.shape
        #print('char',char_sequence.shape)
        input_embed=self.decoderCharEmb(char_sequence[0:length-1])
        outputs,(h_n,c_n)=self.charDecoder(input_embed,dec_hidden)
        scores=self.char_output_projection(outputs)
        _,_,vocab_size=scores.shape
        #print('scores',scores.shape)
        scores=scores.reshape((length-1)*batch,vocab_size)
        #print('scores',scores.shape)
        crossloss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'])
        target=char_sequence[1:length]
        target=target.reshape((length-1)*batch)
        #print('target',target.shape)
        loss=crossloss(scores,target)
        
        return loss
        

        
        


        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        hprev,cprev=initialStates
        _,batch,_=hprev.shape
        softmax = nn.Softmax(dim=1)
        #output_words=[[]]
        output_words=['{']*batch 
        current_chars=(self.target_vocab.char2id['{'])*torch.ones((1,batch),dtype=torch.long,device=device)

        current_chars=self.decoderCharEmb(current_chars)
        #print('current_char',current_chars)
        for t in range(max_length-1):
           outputs,(hnow,cnow)=self.charDecoder(current_chars,(hprev,cprev))
           scores=self.char_output_projection(outputs)
           _,_,vocab_size=scores.shape
           scores=scores.reshape(batch,vocab_size)
           prob=softmax(scores)
           #print('prob',prob)
           arg=torch.argmax(prob, dim=1)
           #print('argall',arg)
           #current_chars=torch.zeros((1,batch),dtype=torch.long,device=device)
           #current_chars_str=[None]*batch
           current_chars=torch.zeros((1,batch),dtype=torch.long,device=device)
           for i in range(batch): 
              #print('arg',arg[i])
              char=self.target_vocab.id2char[arg[i].item()]
              #current_chars_str[i]=char
              current_chars[0][i]=self.target_vocab.char2id[char]
              output_words[i]+=char
              #arg[i]
           #output_words.append([current_chars_str])
           
        
           #print('out',output_words)
           #current_chars=current_chars.reshape(1,batch)
           current_chars=self.decoderCharEmb(current_chars)
           hprev=hnow
           cprev=cnow
        for i in range(batch):
            output_words[i] = output_words[i][1:]
            output_words[i] = output_words[i].partition('}')[0]
        
        #output_words_trun=[[]*(1)]*batch
        #print('me')
        #print('outbef',output_words)
        
        return output_words
        
         
        
        
        ### END YOUR CODE

