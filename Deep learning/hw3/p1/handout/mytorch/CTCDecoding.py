# import numpy as np

# class GreedySearchDecoder(object):

#     def __init__(self, symbol_set):
#         """
        
#         Initialize instance variables

#         Argument(s)
#         -----------

#         symbol_set [list[str]]:
#             all the symbols (the vocabulary without blank)

#         """

#         self.symbol_set = symbol_set


#     def decode(self, y_probs):
#         """

#         Perform greedy search decoding

#         Input
#         -----

#         y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
#             batch size for part 1 will remain 1, but if you plan to use your
#             implementation for part 2 you need to incorporate batch_size

#         Returns
#         -------

#         decoded_path [str]:
#             compressed symbol sequence i.e. without blanks or repeated symbols

#         path_prob [float]:
#             forward probability of the greedy path

#         """

#         decoded_path = []
#         blank = 0
#         path_prob = 1

#         y_probs = y_probs.squeeze()
#         T = y_probs.shape[1]
#         print("y_probs", y_probs.shape)
#         for time in range(T):
#             t_probs = y_probs[:,time]
#             idx = np.argmax(t_probs)
            
#             # save
#             decoded_path.append(idx)
#             path_prob *= t_probs[idx]

#         # TODO:
#         # 1. Iterate over sequence length - len(y_probs[0])
#         # 2. Iterate over symbol probabilities
#         # 3. update path probability, by multiplying with the current max probability
#         # 4. Select most probable symbol and append to decoded_path
#         # 5. Compress sequence (Inside or outside the loop)

#         ans = " "
#         for token in decoded_path:
#             if token == 0:
#                 continue
#             letter = self.symbol_set[token-1]
#             if ans[-1] == letter:
#                 continue
#             else:
#                 ans += letter
#         return ans[1:], path_prob


# class BeamSearchDecoder(object):

#     def __init__(self, symbol_set, beam_width):
#         """

#         Initialize instance variables

#         Argument(s)
#         -----------

#         symbol_set [list[str]]:
#             all the symbols (the vocabulary without blank)

#         beam_width [int]:
#             beam width for selecting top-k hypotheses for expansion

#         """

#         self.symbol_set = symbol_set
#         self.beam_width = beam_width

#     def scale(self,x:dict, top_k):
#         return dict(sorted(x.items(), key = lambda item : item[1], reverse = True)[:top_k])


#     def decode(self, y_probs):
#         """
        
#         Perform beam search decoding

#         Input
#         -----

#         y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
# 			batch size for part 1 will remain 1, but if you plan to use your
# 			implementation for part 2 you need to incorporate batch_size

#         Returns
#         -------
        
#         forward_path [str]:
#             the symbol sequence with the best path score (forward probability)

#         merged_path_scores [dict]:
#             all the final merged paths with their scores

#         """

#         T = y_probs.shape[1]
#         y_probs = y_probs.squeeze()
#         print("y_probs", y_probs.shape, y_probs)
#         print()
        

#         scores = {}
#         for time in range(T):
#             print("time", time)
#             t_probs = y_probs[:,time]
#             if time == 0:
#                 for i, prob in enumerate(t_probs):
#                     if i == 0:
#                         scores["-"] = prob
#                     else:
#                         scores[self.symbol_set[i-1]] = prob
#                 scores = self.scale(scores, self.beam_width)
#             else:
#                 blank_dict = {}
#                 letter_dict = {}
#                 for path, path_score in scores.items():
#                     for i, i_prob in enumerate(t_probs):
#                         if i == 0:
#                             blank_dict[path] = path_score * i_prob
#                         else:
#                             letter = self.symbol_set[i-1]
#                             if letter == path[-1]:
#                                 if path not in letter_dict:
#                                     letter_dict[path] = path_score * i_prob
#                                 else:
#                                     letter_dict[path] += path_score * i_prob
#                             else:
#                                 if path == "-":
#                                     new_path = letter
#                                 else:
#                                     new_path = path + letter
#                                 if new_path not in letter_dict:
#                                     letter_dict[new_path] = path_score * i_prob
#                                 else:
#                                     letter_dict[new_path] += path_score * i_prob
                
#                 merge = {}
#                 for key,value in blank_dict.items():
#                     merge[key] = value
#                 for key,value in letter_dict.items():
#                     if key in merge:
#                         merge[key] += value
#                     else:
#                         merge[key] = value
#                 scores = self.scale(merge, self.beam_width)
               
#                 # print("blank_dict", blank_dict.items())
#                 # print("letter_dict", letter_dict.items())
#                 # print("merge", merge.items())
        

               


            
#             # 

                
           
            
  
        

#         bestPath, FinalPathScore = None, {"a":0}
        
        
#         return bestPath, FinalPathScore #cabacb, dict
#         #raise NotImplementedError


# edited by sumin

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

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)

        n_iter = y_probs.shape[1]

        path = []
        for i in range(n_iter):
            max_prob = 0
            symbol = ''

            probs = y_probs[:, i, :]
            path_prob *= probs.max()

            i_sym = probs.argmax()
            if i_sym != blank:
                path.append(self.symbol_set[i_sym-1])

        decoded_path.append(path[0])
        for i in range(1, len(path)):
            if path[i-1] != path[i]:
                decoded_path.append(path[i])

        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob
        # raise NotImplementedError


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
        bestPath = []

        symbol_sets = [' '] + self.symbol_set
        n_iter = y_probs.shape[1]

        bestPath, FinalPathScore = None, None

        # initialize
        blank_path = {' ': y_probs[0, 0, 0]}
        symbol_path = {self.symbol_set[i]: y_probs[i+1, 0, 0]
                       for i in range(len(self.symbol_set))}

        score_list = list(blank_path.items()) + list(symbol_path.items())

        for t in range(1, T):
            print('t:', t)
            print('score_list:', score_list)
            # prune
            sorted_list = sorted(score_list, key=lambda x: x[1], reverse=True)
            print('sorted_list:', sorted_list)
            sorted_list = sorted_list[:self.beam_width]
            print('sorted_list:', sorted_list)
            cutoff = sorted_list[-1][-1]
            print('cutoff:', cutoff)

            blank_path = {k: v for k, v in blank_path.items() if v >= cutoff}
            symbol_path = {k: v for k, v in symbol_path.items() if v >= cutoff}

            print('blank_path:', blank_path)
            print('symbol_Path:', symbol_path)
            print('len(blank_path):', len(blank_path))
            print('len(symbol_path):', len(symbol_path))
            print()

            curr_prob_blank = y_probs[0, t, 0]
            curr_prob_symb = y_probs[1:, t, 0]

            # extend_with_blanks
            temp_blank_path = {}
            for k, v in blank_path.items():
                temp_blank_path[k] = v * curr_prob_blank

            for k, v in symbol_path.items():
                if k in temp_blank_path.keys():
                    temp_blank_path[k] += v * curr_prob_blank
                else:
                    temp_blank_path[k] = v * curr_prob_blank

            # extend_with_symbols
            temp_symbol_path = {}
            for idx, symb in enumerate(self.symbol_set):
                for k, v in blank_path.items():
                    if k == ' ':
                        new_key = symb
                    else:
                        new_key = k + symb

                    new_key = new_key.strip()
                    new_val = v * curr_prob_symb[idx]

                    if new_key not in temp_symbol_path:
                        temp_symbol_path[new_key] = new_val 
                    else:
                        temp_symbol_path[new_key] += new_val

                for k, v in symbol_path.items():
                    if k == ' ':
                        new_key = symb
                    elif k[-1] != symb:
                        new_key = k + symb
                    else:
                        new_key = k

                    new_key = new_key.strip()
                    new_val = v * curr_prob_symb[idx]

                    if new_key not in temp_symbol_path:
                        temp_symbol_path[new_key] = new_val 
                    else:
                        temp_symbol_path[new_key] += new_val

            blank_path = temp_blank_path.copy()
            symbol_path = temp_symbol_path.copy()

            score_list = list(blank_path.items()) + list(symbol_path.items())

            print('curr_prob_blank:', curr_prob_blank)
            print('curr_prob_symb:', curr_prob_symb)
            print('blank_path:', blank_path)
            print('symbol_Path:', symbol_path)
            print('len(blank_path):', len(blank_path))
            print('len(symbol_path):', len(symbol_path))
            print()

        merged_path = blank_path.copy()
        for k, v in symbol_path.items():
            if k not in merged_path.keys():
                merged_path[k] = v
            else:
                merged_path[k] += v

        FinalPathScore = merged_path.copy()
        bestPath = max(merged_path, key=FinalPathScore.get)

        print('FinalPathScore:', FinalPathScore)
        print('bestPath:', bestPath)
        return bestPath, FinalPathScore
        #raise NotImplementedError

