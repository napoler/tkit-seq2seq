from transformers import BertTokenizerFast
# from sklearn.feature_extraction.text import CountVectorizer
import csv
import random

# 简单的创建 加法数据
tokenizer = BertTokenizerFast.from_pretrained("tokenizer")
with open("data/demo.csv", 'w') as f:
    w = csv.writer(f)
    for i in range(10000):
        print(i)
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)

        c = a + b
        sent1 = str(a) + "+" + str(b)
        w.writerow([sent1, str(c)])

        c = a - b
        sent1 = str(a) + "-" + str(b)
        w.writerow([sent1, str(c)])

        c = a * b
        sent1 = str(a) + "*" + str(b)
        w.writerow([sent1, str(c)])

        c = a / b
        sent1 = str(a) + "/" + str(b)
        w.writerow([sent1, str(c)])
