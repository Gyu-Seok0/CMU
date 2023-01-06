import numpy as np
from activation import *


class RNNCell(object):
    """RNN Cell class."""

    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """
        RNN Cell forward (single time step).

        Input (see writeup for explanation)
        -----
        x: (batch_size, input_size)
            input at the current time step

        h_prev_t: (batch_size, hidden_size)
            hidden state at the previous time step and current layer

        Returns
        -------
        h_t: (batch_size, hidden_size)
            hidden state at the current time step and current layer

        batch_size = 2
        input_size = 2
        hidden_size = 3
        h_t shape: 2, 3
        """

        """
        ht = tanh(Wihxt + bih + Whhht−1 + bhh) 
        self.W_ih.shape: (3, 2)
        self.b_ih.shape: (3,)
        self.W_hh.shape: (3, 3)
        self.b_hh.shape: (3,)
        x_.shape: (2, 2)
        h_prev_t_.shape: (2, 3)
        """
        affine_input = x @ self.W_ih.T + self.b_ih
        affine_hidden = h_prev_t @ self.W_hh.T + self.b_hh
        affine_final = affine_input + affine_hidden 
        h_t = self.activation(affine_final)

        return h_t

    def backward(self, delta, h_t, h_prev_l, h_prev_t):
        """
        RNN Cell backward (single time step).

        Input (see writeup for explanation)
        -----
        delta: (batch_size, hidden_size)
                Gradient w.r.t the current hidden layer

        h_t: (batch_size, hidden_size)
            Hidden state of the current time step and the current layer

        h_prev_l: (batch_size, input_size): 3, 10
                    Hidden state at the current time step and previous layer

        h_prev_t: (batch_size, hidden_size): 3, 20
                    Hidden state at previous time step and current layer

        dz.shape: (3, 20)
        input_size: 10
        hidden_size: 20

        Returns
        -------
        dx: (batch_size, input_size)
            Derivative w.r.t.  the current time step and previous layer

        dh_prev_t: (batch_size, hidden_size)
            Derivative w.r.t.  the previous time step and current layer

        """
        batch_size = delta.shape[0]
        # 0) Done! Step backward through the tanh activation function.
        # Note, because of BPTT, we had to externally save the tanh state, and
        # have modified the tanh activation function to accept an optionally input.
        dz = delta * (1 - h_t**2)# TODO

        # 1) Compute the averaged gradients of the weights and biases
        self.dW_ih += (dz.T @ h_prev_l) / batch_size # TODO, 20, 10
        self.dW_hh += (dz.T @ h_prev_t) / batch_size # TODO, 20, 20
        self.db_ih += dz.mean(axis=0) # TODO, 20
        self.db_hh += dz.mean(axis=0) # TODO, 

        # # 2) Compute dx, dh_prev_t
        dx        = dz @ self.W_ih # TODO, 3, 10
        dh_prev_t = dz @ self.W_hh # TODO, 3, 20

        # 3) Return dx, dh_prev_t
        return dx, dh_prev_t
        #raise NotImplementedError
