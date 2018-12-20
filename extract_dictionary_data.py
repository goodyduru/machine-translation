import os
import random
import re
from bs4 import BeautifulSoup

DIR = "./data/dict/"
DATA_DIR = "./data/"
all_files = os.listdir(DIR)
all_files = [file_name for file_name in all_files if file_name.find("html") > 0]
english_words = []
igbo_words = []
def find_main_table(table_list):
    main_table = ""
    for table_element in table_list:
        table_text = table_element.get_text().lower()
        if len(re.findall("continuous\s+meaning", table_text)) > 0:
            main_table = table_element
            break
    return main_table

def clean_up_sentences(sentence):
    clean_sentence = re.sub('\{.*\}', "", sentence)
    clean_sentence = re.sub('\(.*\)', "", clean_sentence)
    clean_sentence = re.sub('\[.*\]', "", clean_sentence)
    return clean_sentence

def process_sentence(sentence):
    comma_separated = re.split('[=;!,]', sentence)
    final_words = []
    for word in comma_separated:
        if word.strip() != "":
            final_words.append(word.strip())
    return final_words

def process_both_sentences(english_sentence, igbo_sentence):
    english = process_sentence(english_sentence)
    igbo = process_sentence(igbo_sentence)
    for english_word in english:
        if english_word.find('\n') > -1:
            continue
        for igbo_word in igbo:
            english_words.append(english_word)
            igbo_words.append(igbo_word)

def get_both_english_and_igbo_words(igbo_words_elements):
    for igbo_word_element in igbo_words_elements:
        igbo_sentence = igbo_word_element.get_text()
        table_row = igbo_word_element.parent.parent.parent.parent
        table_cells = table_row.find_all('td')
        if len(table_cells) < 3:
           continue
        english_cell = table_cells[2]
        english_span = english_cell.find('span')
        english_sentence = english_span.get_text()
        english_sentence = clean_up_sentences(english_sentence)
        igbo_sentence = clean_up_sentences(igbo_sentence)
        process_both_sentences(english_sentence, igbo_sentence)

for file_name in all_files:
    with open(DIR + file_name, encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")
        table_elements_list = soup.find_all('table')
        table_element = find_main_table(table_elements_list)
        igbo_words_elements = table_element.find_all('b')
        get_both_english_and_igbo_words(igbo_words_elements)


with open(DATA_DIR+"igbo_dict.txt", "w+", encoding="utf-8") as igbo_file:
    string = "\n".join(igbo_words)
    igbo_file.write(string)

with open(DATA_DIR+"english_dict.txt", "w+", encoding="utf-8") as english_file:
    string = "\n".join(english_words)
    english_file.write(string)