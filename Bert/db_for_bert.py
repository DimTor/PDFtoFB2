import pandas as pd

with open('../../Spell/my_big_txt.txt', 'r', encoding='utf-8') as db:
    data = db.read()

date_mas = data.split()
new_mas = []
for i in range(len(date_mas) // 200 -1):
    new_mas.append(' '.join(date_mas[i*100:(i+1)*100]))
print(new_mas)
my_dict = {'text': new_mas}
my_df = pd.DataFrame(my_dict)
print(my_df.head())
print(my_df)
my_df.to_csv('out.csv', encoding='utf-8')