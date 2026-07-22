import string


class CTCTextEncoder:
    def __init__(self):
        alphabet = list(string.ascii_lowercase + " ")

        vocab = ["<blank>"] + alphabet
        
        self.char2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        
    def encode(self, text):
        tokens = [self.char2idx[char] for char in text if char in self.char2idx]
        return tokens    

    def decode(self, tokens):
        return "".join([self.idx2char[token] for token in tokens])
    
    def ctc_decode(self, tokens, blank=0):
        result = []
        prev = blank

        for t in tokens:
            t = t.item()

            if t != prev and t != blank:
                result.append(t)

            prev = t

        return result
