import numpy as np


class CTC(object):
    def __init__(self, BLANK=0):
        # No need to modify
        self.BLANK = BLANK
        a =1
        self.alpha = None
        self.beta = None

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)

        N = len(extended_symbols)

        # -------------------------------------------->
        # TODO
        skip_connect = []
        before_word = None
        for i, word in enumerate(extended_symbols):
            if word == self.BLANK:
                skip_connect.append(0)
            else:
                if before_word is None:
                    skip_connect.append(0)

                elif before_word != word:
                    skip_connect.append(1)

                else:
                    skip_connect.append(0)

                before_word = word

        # <---------------------------------------------

        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect
        #raise NotImplementedError


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S)) #  12 x 5 

        # -------------------------------------------->
        

        # TODO: Intialize alpha[0][0]
        alpha[0][0] = logits[0, extended_symbols[0]]
        # TODO: Intialize alpha[0][1]
        alpha[0][1] = logits[0, extended_symbols[1]]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
       
        for time in range(1, T, 1): # 12
            for sym in range(S): # 5
                if sym == 0:
                    alpha[time][sym] = alpha[time-1][sym]

                elif extended_symbols[sym] == 0: # if blank
                    alpha[time][sym] = sum(alpha[time-1][sym - 1 : sym + 1])
               
                else: # if sys
                    if sym < 2:
                        alpha[time][sym] = sum(alpha[time-1][sym - 1 : sym + 1])
                    else:
                        if skip_connect[sym] == 1:
                            alpha[time][sym] = sum(alpha[time-1][sym - 2 : sym + 1]) 
                        elif skip_connect[sym - 2] == 1:
                            if skip_connect[sym - 1] == 1:
                                alpha[time][sym] = sum(alpha[time-1][sym]) 
                            else:
                                alpha[time][sym] = sum(alpha[time-1][sym - 1 : sym + 1])
                        else:
                            alpha[time][sym] = sum(alpha[time-1][sym - 2 : sym + 1]) 
                # product 
                alpha[time][sym] *= logits[time, extended_symbols[sym]]

        # IMP: Remember to check for skipConnect when calculating alpha

        # <---------------------------------------------
        self.alpha = alpha
        return alpha
        #raise NotImplementedError


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S)) # 12 * 5

        # init
        beta[T-1][S-1] = 1.0 #logits[T-1, extended_symbols[S-1]]
        beta[T-1][S-2] = 1.0 #logits[T-1, extended_symbols[S-2]]
        
        # -------------------------------------------->
        # TODO
        for time in range(T-2,-1,-1):
            for sym in range(S-1, -1, -1):
                if sym == S-1:
                    beta[time][sym] = beta[time+1][sym] * logits[time+1, extended_symbols[sym]]
                elif extended_symbols[sym] == 0:
                    beta[time][sym] = beta[time+1][sym] * logits[time+1, extended_symbols[sym]] + beta[time+1][sym+1] * logits[time+1, extended_symbols[sym+1]] 
                else:
                    if sym + 2 < S:
                        if skip_connect[sym] == 1:
                            if skip_connect[sym + 2] == 1:
                                beta[time][sym] = beta[time+1][sym] * logits[time+1, extended_symbols[sym]] + beta[time+1][sym+1] * logits[time+1, extended_symbols[sym+1]] + beta[time+1][sym+2] * logits[time+1, extended_symbols[sym+2]] 
                            else:
                                beta[time][sym] = beta[time+1][sym] * logits[time+1, extended_symbols[sym]] + beta[time+1][sym+1] * logits[time+1, extended_symbols[sym+1]]
                        else:
                            beta[time][sym] = beta[time+1][sym] * logits[time+1, extended_symbols[sym]] + beta[time+1][sym+1] * logits[time+1, extended_symbols[sym+1]] + beta[time+1][sym+2] * logits[time+1, extended_symbols[sym+2]] 
                    else:
                        beta[time][sym] = beta[time+1][sym] * logits[time+1, extended_symbols[sym]] + beta[time+1][sym+1] * logits[time+1, extended_symbols[sym+1]] 
                
        # <--------------------------------------------

        return beta
        #raise NotImplementedError
        

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        row_sum = np.sum(alpha * beta, axis = 1)
        row_sum = row_sum[:,None]
        # <---------------------------------------------
        gamma = (alpha * beta) / row_sum
        return gamma
        #raise NotImplementedError


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        self.batch_logits = []
        
        # <---------------------------------------------
    def softmax(self, x):
        #return np.exp(array) / np.exp(array).sum()
        return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """
        #print("logits", logits.shape) # 15(time) * 12(batch) * 8(symbol)
        #print("target",target)
        #print("input_lengths",input_lengths )
        #print("target_lengths", target_lengths)
        # No need to modify
        self.logits = logits # time * batch * sym
        self.target = target # batch * padded_tagrget
        self.input_lengths = input_lengths # batch
        self.target_lengths = target_lengths # batch

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            b_target_len = target_lengths[batch_itr]
            b_target = target[batch_itr,:b_target_len]

            #     Truncate the logits to input length
            b_input_len = input_lengths[batch_itr]
            b_logits = logits[:b_input_len, batch_itr, :]
            self.batch_logits.append(b_logits)
            

            #     Extend target sequence with blank
            #print("b_target",b_target)
            b_extend, b_skip =  self.ctc.extend_target_with_blank(b_target)
            #print("b_extend", b_extend)
            self.extended_symbols.append(b_extend)

            #print("b_extend", b_extend, b_extend.shape)
            #print("b_logits", b_logits, b_logits.shape)

            #     Compute forward probabilities
            b_alpha = self.ctc.get_forward_probs(b_logits, b_extend, b_skip)
            #     Compute backward probabilities
            b_beta = self.ctc.get_backward_probs(b_logits, b_extend, b_skip)
            #     Compute posteriors using total probability function
            b_gamma = self.ctc.get_posterior_probs(b_alpha, b_beta) # input * (2*target_len + 1)
            self.gammas.append(b_gamma)
            #     Compute expected divergence for each batch and store it in totalLoss

            b_loss = 0
            for t in range(b_input_len):
                for r in range(2*b_target_len + 1):
                    b_loss -= b_gamma[t, r] * np.log(b_logits[t, b_extend[r]]) 
            total_loss[batch_itr] = b_loss
        total_loss = np.sum(total_loss) / B
        return total_loss

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            for t in range(self.input_lengths[batch_itr]):
                for s in range(len(self.extended_symbols[batch_itr])):
                    a = self.gammas[batch_itr][t][s]
                    b = self.batch_logits[batch_itr][t][self.extended_symbols[batch_itr][s]] 
                    dY[t][batch_itr][self.extended_symbols[batch_itr][s]] -= a/b 
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            

        return dY
