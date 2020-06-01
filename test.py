
import re



a = '123dsahbjk123nkjdsahd123nkjshfkd123'

for i in re.finditer('(\d+)', a):
    print(i.start(), i.end(), i.group())