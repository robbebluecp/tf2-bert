import os
from keras_bert.util_tools.download import Downloader


base_dir = os.getcwd()


class Tokenizer:

    def __init__(self,
                 file_name: str = base_dir + '/data/chinese_L-12_H-768_A-12/vocab.txt',
                 type=''):
        self.type = type
        self.check_download(name='bert')
        self.token_dict, self.token_dict_inv = self.get_token_dict(file_name)

    @staticmethod
    def get_token_dict(file_name):
        token_dict = {}
        with open(file_name, 'r') as f:
            words = f.readlines()
            for word in words:
                word = word.rstrip()
                if word not in token_dict:
                    token_dict[word] = len(token_dict)
            f.close()
        token_dict_inv = {token_dict[key]: key for key in token_dict}
        return token_dict, token_dict_inv

    @staticmethod
    def check_download(mode: int=1, name: str = 'bert'):
        Downloader(mode=mode)(name)

    def tokenize(self, text, sencond_text=None, max_len=512):
        return [x + [0] * (max_len - len(x)) for x in self.make_vectors(text, sencond_text, max_len)]


    def make_vectors(self, text, sencond_text=None, max_len=512):
        text = text.replace('[MASK]', ' ')
        token_words = ['[CLS]']
        token_array = [self.token_dict['[CLS]']]
        mask_array = [1] if self.type.lower() == 'cls' else [0]
        for word in text[:max_len-2]:
            if word == ' ':
                word = '[MASK]'
            token_words.append(word)
            if word not in self.token_dict:
                token_array.append(100)
            else:
                token_array.append(self.token_dict[word])
            mask_array.append(1 if word == '[MASK]' or self.type.lower() == 'cls' else 0)
        token_words.append('[SEP]')
        token_array.append(102)
        mask_array.append(1 if self.type.lower() == 'cls' else 0)
        segment_array = [0] * len(token_array)
        if sencond_text:
            second_token_words, second_token_array, second_segment_array, second_mask_array = self.make_vectors(text=sencond_text, max_len=max_len)
            token_words += second_token_words[1:]
            token_array += second_token_array[1:]
            segment_array += [1] * (len(second_segment_array) - 1)
            mask_array += second_mask_array[1:]
        return [token_words, token_array, segment_array, mask_array]



