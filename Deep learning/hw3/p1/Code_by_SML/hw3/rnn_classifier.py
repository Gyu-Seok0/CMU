import numpy as np
import sys

sys.path.append("mytorch")
from rnn_cell import *
from linear import *


class RNNPhonemeClassifier(object):
    """RNN Phoneme Classifier class."""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO: Understand then uncomment this code :)
        self.rnn = [
            RNNCell(input_size, hidden_size) if i == 0 
                else RNNCell(hidden_size, hidden_size)
                    for i in range(num_layers)
        ]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """Initialize weights.

        Parameters
        ----------
        rnn_weights:
                    [
                        [W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
                        [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1],
                        ...
                    ]

        linear_weights:
                        [W, b]

        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.W = linear_weights[0]
        self.output_layer.b = linear_weights[1].reshape(-1, 1)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):
        """RNN forward, multiple layers, multiple time steps.

        Parameters
        ----------
        x: (batch_size, seq_len, input_size), 5, 10, 40
            Input

        h_0: (num_layers, batch_size, hidden_size) 2, 5, 32
            Initial hidden states. Defaults to zeros if not specified

        self.x.shape: (5, 10, 40)
        hidden.shape: (2, 5, 32)
        self.output_layer.W.shape: (138, 32)
        self.output_layer.b.shape: (138, 1)

        x[:, i_seq, :].shape: (5, 40)
        hidden[i_l-1].shape: (5, 32)

        Returns
        -------
        logits: (batch_size, output_size) 5, 138

        Output (y): logits

        """
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())
        logits = self.output_layer(hidden[-1, :, :])

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
        #   Iterate over the length of your self.rnn (through the layers)
        #       Run the rnn cell with the correct parameters and update
        #       the parameters as needed. Update hidden.
        #   Similar to above, append a copy of the current hidden array to the hiddens list

        
        # TODO
        # print(self.hiddens[0].shape): (2, 5, 32)
        for i_seq in range(seq_len):
            new_hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
            for i_l in range(self.num_layers):
                # return self.forward(x, h_prev_t)
                if i_l == 0:
                    new_hidden[i_l, :, :] = self.rnn[i_l](x[:, i_seq, :], 
                                                          self.hiddens[-1][i_l, :, :])
                else:
                    new_hidden[i_l, :, :] = self.rnn[i_l](new_hidden[i_l-1, :, :],
                                                          self.hiddens[-1][i_l, :, :])

            hidden = new_hidden
            self.hiddens.append(hidden.copy())

        # Get the outputs from the last time step using the linear layer and return it
        # <--------------------------
        logits = self.output_layer(self.hiddens[-1][-1, :, :])
        return logits
        #raise NotImplementedError

    def backward(self, delta):
        """RNN Back Propagation Through Time (BPTT).

        Parameters
        ----------
        delta: (batch_size, hidden_size)

        gradient: dY(seq_len-1)
                gradient w.r.t. the last time step output.

        Returns
        -------
        dh_0: (num_layers, batch_size, hidden_size)

        gradient w.r.t. the initial hidden states

        """
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        """

        * Notes:
        * More specific pseudocode may exist in lecture slides and a visualization
          exists in the writeup.
        * WATCH out for off by 1 errors due to implementation decisions.

        Pseudocode:
        * Iterate in reverse order of time (from seq_len-1 to 0)
            * Iterate in reverse order of layers (from num_layers-1 to 0)
                * Get h_prev_l either from hiddens or x depending on the layer
                    (Recall that hiddens has an extra initial hidden state)
                * Use dh and hiddens to get the other parameters for the backward method
                    (Recall that hiddens has an extra initial hidden state)
                * Update dh with the new dh from the backward pass of the rnn cell
                * If you aren't at the first layer, you will want to add
                  dx to the gradient from l-1th layer.

        * Normalize dh by batch_size since initial hidden states are also treated
          as parameters of the network (divide by batch size)

        Tip: You may or may not require += at places. Think about it and code

        def backward(self, delta, h_t, h_prev_l, h_prev_t):
        return dx, dh_prev_t
        
        """
        # TODO

        for i_seq in range(seq_len-1, -1, -1):
            for i_l in range(self.num_layers-1, -1, -1):
                if i_l != 0:
                    dx, dh_prev_t = self.rnn[i_l].backward(dh[i_l], 
                                                           self.hiddens[i_seq+1][i_l, :, :],
                                                           self.hiddens[i_seq+1][i_l-1, :, :], 
                                                           self.hiddens[i_seq][i_l, :, :])

                else:
                    dx, dh_prev_t = self.rnn[i_l].backward(dh[i_l], 
                                                           self.hiddens[i_seq+1][i_l, :, :],
                                                           self.x[:, i_seq, :],
                                                           self.hiddens[i_seq][i_l, :, :])

                dh[i_l] = dh_prev_t

                if (i_seq == seq_len-1) and (i_l > 0):
                    dh[i_l-1] = dx
                elif (i_seq < seq_len-1) and (i_l > 0):
                    dh[i_l-1] += dx


        return dh / batch_size
