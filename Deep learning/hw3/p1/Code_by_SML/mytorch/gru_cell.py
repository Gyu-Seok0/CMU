import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        z_t = ((self.Wzx @ self.x + self.bzx) + (self.Wzh @ self.hidden + self.bzh)) 
        self.z = self.z_act(z_t)

        r_t = ((self.Wrx @ self.x + self.brx) + (self.Wrh @ self.hidden + self.brh))
        self.r = self.r_act(r_t)

        n_t = ((self.Wnx @ self.x + self.bnx) + (self.Wnh @ self.hidden + self.bnh) * self.r)
        self.n = self.h_act(n_t)

        h_t = (1 - self.z) * self.n + self.z * self.hidden 
        
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """

        dLdn = (1 - self.z) * delta 
        dLdz = (self.hidden - self.n) * delta 
        dLdr = self.h_act.backward() * (self.Wnh @ self.hidden + self.bnh) * dLdn

        dLdn = dLdn * self.h_act.backward(self.n)
        dLdz = dLdz * self.z_act.backward()
        dLdr = dLdr * self.r_act.backward()

        x_reshape = self.x.reshape((self.d, 1))
        hidden_reshape = self.hidden.reshape((self.h, 1))

        self.dWrx += (x_reshape @ dLdr).T
        self.dWzx += (x_reshape @ dLdz).T
        self.dWnx += (x_reshape @ dLdn).T

        self.dWrh += (hidden_reshape @ dLdr).T 
        self.dWzh += (hidden_reshape @ dLdz).T
        self.dWnh += (hidden_reshape @ (dLdn * self.r)).T

        self.dbrx += dLdr.squeeze()
        self.dbzx += dLdz.squeeze()
        self.dbnx += dLdn.squeeze()

        self.dbrh += dLdr.squeeze()
        self.dbzh += dLdz.squeeze()
        self.dbnh += (dLdn * self.r).squeeze()

        dx = (dLdr @ self.Wrx) + (dLdz @ self.Wzx) + (dLdn @ self.Wnx)

        dh_prev_t = (delta * self.z 
                     + (dLdr @ self.Wrh) 
                     + (dLdz @ self.Wzh) 
                     + ((dLdn * self.r) @ self.Wnh))
        
        
        assert dx.shape == (1, self.d)
        assert dh_prev_t.shape == (1, self.h)

        return dx, dh_prev_t
