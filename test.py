
import re


sentence = '张三李四和王五都在广东中国人寿上班班'
array = [1, 2, 1, 2, 0, 1, 2, 0, 0, 5, 6, 6, 6, 6, 6, 0, 0]

array_str = '12120120056666600'
label_ints_dic_rever = {'12': 'PER', '34': 'LOC', '56': 'ORG'}

result = {}
for i in label_ints_dic_rever:
    re_iter = re.finditer('(%s+)' % i, array_str)
    des = label_ints_dic_rever[i]
    if des not in result:
        result[des] = []
    for j in re_iter:
        result[des].append(sentence[j.start(): j.end()])


for i in result:
    if result[i]:
        print(i, ':', *result[i])