import pandas as pd
import numpy as np

def convert_file_into_array(file_name):
    file_list = []
    with open(file_name, "r", encoding="utf-8") as file:
        file_content = file.read()
        file_list = file_content.split('\n')
    return file_list

english_dict = convert_file_into_array('../data/english-dictionary.txt')
igbo_dict = convert_file_into_array('../data/igbo-dictionary.txt')
english_verses = convert_file_into_array('../data/english-verses.txt')
igbo_verses = convert_file_into_array('../data/igbo-verses.txt')

english_dict.extend(english_verses)
igbo_dict.extend(igbo_verses)

english = np.array(english_dict)
igbo = np.array(igbo_dict)

pd_dict = {'en': english, 'ig': igbo}
pd_obj = pd.DataFrame(data=pd_dict)

final = pd_obj.sample(frac=1).reset_index(drop=True)
final.to_csv('../data/data.csv')