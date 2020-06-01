import platform

p_name = platform.platform().lower()
if p_name.startswith('darwin'):
    bert_path = '/Users/robbe/others/tf_data/chinese_L-12_H-768_A-12'
elif p_name.startswith('linux'):
    bert_path = '/opt/chinese_L-12_H-768_A-12'
else:
    bert_path = ''
