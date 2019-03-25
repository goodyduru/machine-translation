import re

#This script converts all the chapters of the english and igbo bible text to verses.

DATA_DIR = "../data/"
english_verses = []
igbo_verses = []

def convert_file_into_array(file_name):
    file_list = []
    with open(file_name, "r", encoding="utf-8") as file:
        for line in file:
            file_list.append(line)
    return file_list

english_chapters = convert_file_into_array(DATA_DIR+"eng.txt")
igbo_chapters = convert_file_into_array(DATA_DIR+"igbo.txt")

def get_verses(first_chapters, second_chapters):
    first_verses = []
    second_verses = []
    for first, second in zip(first_chapters, second_chapters):
        first_list = re.split("(\(\d+\))", first)
        second_list = re.split("(\(\d+\))", second)
        if len(first_list) != len(second_list):
            continue
        first_verses.extend(get_verse(first_list))
        second_verses.extend(get_verse(second_list))
    return first_verses, second_verses

def get_verse(chapter):
    begin = ""
    verse = []
    for word in chapter:
        if word.strip() == "":
            continue
        if len(word) <= 6:
            word = word.replace("(", "")
            begin = word.replace(")", "")
        else:
            begin += word.replace("\n", "")
            verse.append(begin.strip())
    return verse
        

english_verses, igbo_verses = get_verses(english_chapters, igbo_chapters)

def save_file(array, file_name):
    with open(DATA_DIR+file_name, "w+", encoding="utf-8") as file:
        joined_array = "\n".join(array)
        file.write(joined_array)


save_file(english_verses, "english-verses.txt")
save_file(igbo_verses, "igbo-verses.txt")