import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        y_probs = y_probs.squeeze()
        T = y_probs.shape[1]
        print("y_probs", y_probs.shape)
        for time in range(T):
            t_probs = y_probs[:,time]
            idx = np.argmax(t_probs)
            
            # save
            decoded_path.append(idx)
            path_prob *= t_probs[idx]

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        ans = " "
        for token in decoded_path:
            if token == 0:
                continue
            letter = self.symbol_set[token-1]
            if ans[-1] == letter:
                continue
            else:
                ans += letter
        return ans[1:], path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def scale(self,x:dict, top_k):
        return dict(sorted(x.items(), key = lambda item : item[1], reverse = True)[:top_k])


    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        y_probs = y_probs.squeeze()
        print("y_probs", y_probs.shape, y_probs)
        print()
        

        scores = {}
        for time in range(T):
            print("time", time)
            t_probs = y_probs[:,time]
            if time == 0:
                for i, prob in enumerate(t_probs):
                    if i == 0:
                        scores["-"] = prob
                    else:
                        scores[self.symbol_set[i-1]] = prob
                scores = self.scale(scores, self.beam_width)
            else:
                blank_dict = {}
                letter_dict = {}
                for path, path_score in scores.items():
                    for i, i_prob in enumerate(t_probs):
                        if i == 0:
                            blank_dict[path] = path_score * i_prob
                        else:
                            letter = self.symbol_set[i-1]
                            if letter == path[-1]:
                                if path not in letter_dict:
                                    letter_dict[path] = path_score * i_prob
                                else:
                                    letter_dict[path] += path_score * i_prob
                            else:
                                if path == "-":
                                    new_path = letter
                                else:
                                    new_path = path + letter
                                if new_path not in letter_dict:
                                    letter_dict[new_path] = path_score * i_prob
                                else:
                                    letter_dict[new_path] += path_score * i_prob
                
                merge = {}
                for key,value in blank_dict.items():
                    merge[key] = value
                for key,value in letter_dict.items():
                    if key in merge:
                        merge[key] += value
                    else:
                        merge[key] = value
                scores = self.scale(merge, self.beam_width)
               
                # print("blank_dict", blank_dict.items())
                # print("letter_dict", letter_dict.items())
                # print("merge", merge.items())
        

               


            
            # 

                
           
            
  
        

        bestPath, FinalPathScore = None, {"a":0}
        
        
        return bestPath, FinalPathScore #cabacb, dict
        #raise NotImplementedError
