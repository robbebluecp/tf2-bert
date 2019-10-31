import wget
import os
import ssl
import zipfile
import urllib.error

base_dir = os.getcwd()

class Downloader:

    def __init__(self,
                 mode: int = 0):
        self.mode = mode
        self.base_dir = base_dir + '/data/'


    @staticmethod
    def bar(current, total, width, s=set()):
        num = int(current / total * 100)
        if num % 1 == 0 and num not in s:
            s.add(num)
            print('Downloadng: %d%% [%d / %d] bytes ' % (num, current, total))

    @staticmethod
    def download(url, dir, bar):
        try:
            wget.download(url, dir, bar)
        except urllib.error.URLError:
            ssl._create_default_https_context = ssl._create_unverified_context
            wget.download(url, dir, bar)

    @staticmethod
    def unzip(dir, file_name):
        zip_file = zipfile.ZipFile(dir + file_name)
        if file_name in {'chinese_L-12_H-768_A-12.zip', 'news_ner.zip'}:
            zip_file.extractall(path=dir)
        else:
            zip_file.extractall(path=dir + file_name.split('.')[0])

    @staticmethod
    def file_checker(file_path: str):
        if os.path.isfile(file_path):
            return True
        else:
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
        return False

    @staticmethod
    def dir_checker(dir_name: str):
        if dir_name.find('.') >= 0:
            dir_name = dir_name.rsplit('.', -1)[0]
        if os.path.isdir(dir_name):
            return True
        else:
            os.makedirs(dir_name)
            return False

    def distribute_task(self, name):
        if name == 'bert':
            self.download_bert_weights()
        if name == 'hotel_comments':
            self.download_hotel_comments()
        if name == 'news_ner':
            self.download_news_ner()

    def download_bert_weights(self, file_name='chinese_L-12_H-768_A-12.zip'):
        if self.mode == 0:
            if not self.file_checker(self.base_dir + file_name):
                print('prepare to donwload bert...')
                self.download('https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip', self.base_dir, self.bar)
            if not self.dir_checker(self.base_dir + file_name):
                self.unzip(self.base_dir, file_name)

        elif self.mode == 1:
            if not self.file_checker(self.base_dir + file_name):
                print('prepare to donwload bert ')
                self.download('https://lzy-public-data.oss-cn-beijing.aliyuncs.com/chinese_L-12_H-768_A-12.zip', self.base_dir, self.bar)
            if not self.dir_checker(self.base_dir + file_name):
                self.unzip(self.base_dir, file_name)

    def download_hotel_comments(self, file_name='hotel_comments.zip'):
        if not self.file_checker(self.base_dir + file_name):
            print('prepare to donwload hotel comments...')
            self.download('https://lzy-public-data.oss-cn-beijing.aliyuncs.com/hotel_comments.zip', self.base_dir, self.bar)
        if not self.dir_checker(self.base_dir + file_name):
            self.unzip(self.base_dir, file_name)

    def download_news_ner(self, file_name='news_ner.zip'):
        if not self.file_checker(self.base_dir + file_name):
            print('prepare to donwload news NER data...')
            self.download('https://lzy-public-data.oss-cn-beijing.aliyuncs.com/news_ner.zip', self.base_dir, self.bar)
        if not self.dir_checker(self.base_dir + file_name):
            self.unzip(self.base_dir, file_name)

    def __call__(self, name, *args, **kwargs):
        return self.distribute_task(name)
