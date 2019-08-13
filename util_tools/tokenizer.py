import os
from keras_bert.util_tools.download import Downloader


base_dir = os.getcwd()


class Tokenizer:

    def __init__(self,
                 file_name: str = base_dir + '/data/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/vocab.txt'):
        self.check_download('bert')
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
        Downloader(mode)(name)

    def tokenize(self, text, sencond_text=None):
        return [x + [0] * (512 - len(x)) for x in self.make_vectors(text, sencond_text)]

    def make_vectors(self, text, sencond_text=None):
        text = text.replace('[MASK]', ' ')
        token_words = ['[CLS]']
        token_array = [self.token_dict['[CLS]']]
        mask_array = [0]
        for word in text:
            if word == ' ':
                word = '[MASK]'
            token_words.append(word)
            if word not in self.token_dict:
                token_array.append(100)
            else:
                token_array.append(self.token_dict[word])
            mask_array.append(1 if word == '[MASK]' else 0)
        token_words.append('[SEP]')
        token_array.append(102)
        segment_array = [0] * len(token_array)
        if sencond_text:
            second_token_words, second_token_array, second_segment_array, second_mask_array = self.make_vectors(text=sencond_text)
            token_words += second_token_words[1:]
            token_array += second_token_array[1:]
            segment_array += [1] * (len(second_segment_array) - 1)
            mask_array += second_mask_array[1:]
        return [token_words, token_array, segment_array, mask_array]



