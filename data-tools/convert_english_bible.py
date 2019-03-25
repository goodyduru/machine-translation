import os
import re
from bs4 import BeautifulSoup

DIR = "../data/eng/OEBPS/"
DATA_DIR = "../data/"
toc_file =  DIR + "1001061103.xhtml"
books = []
headers = []
html_files = {}
chapter_one_files = []
sentences = []

def replace_some_symbols_with_spaces(word):
    new_word = re.sub("[^A-Za-z0-9]", " ", word)
    return new_word

"""
    This processes the english table of content file
"""
with open(toc_file, encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")
    #Get Title and remove colon
    title = soup.title.string
    headers.append(title)
    #Get Old Testament and New Testament headers
    h2_headers = soup.find_all("h2")
    for header in h2_headers:
        headers.append(replace_some_symbols_with_spaces(header.string.lower().capitalize()))
    #Get Content Headers
    for i in [3, 5, 6]:
        content_header = soup.find(attrs={"data-pid": i})
        headers.append(content_header.string.lower().capitalize())
    #Get Bible Chapters For Old Testament
    for i in range(7, 160, 4):
        book = soup.find(attrs={"data-pid": i})
        books.append(replace_some_symbols_with_spaces(book.string))
        chapter_one_files.append(book.a['href'].split(".")[0])
    #Get Bible Chapters For New Testament
    for i in range(168, 273, 4):
        book = soup.find(attrs={"data-pid": i})
        books.append(replace_some_symbols_with_spaces(book.string))
        chapter_one_files.append(book.a['href'].split(".")[0])

all_files = os.listdir(DIR)
for chapter_one_file in chapter_one_files:
    for file in all_files:
        if file.startswith(chapter_one_file) and file.find('extracted') == -1:
            if chapter_one_file in html_files:
                html_files[chapter_one_file].append(file)
            else:
                html_files[chapter_one_file] = [file]

for chapter_one_file in chapter_one_files:
    for chapter_file in html_files[chapter_one_file]:
        with open(DIR + chapter_file, encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            paragraph_elements = soup.find_all("p", attrs={"class": re.compile("p\d+")})
            chapter_text = ""
            
            for paragraph_element in paragraph_elements:
                classes = paragraph_element['class']

                if len(classes) > 0:
                    strong_elements = paragraph_element.find_all("strong")
                    for strong_element in strong_elements:
                        superscripts = strong_element.findChildren("sup" , recursive=False)
                        if len(superscripts) > 0:
                            superscripts[0].string = "(" + strong_element.get_text().strip() + ") "
                        elif strong_element.next_sibling is not None:
                            if strong_element.next_sibling['id'].startswith('footnote'):
                                strong_element.string = "(" + strong_element.get_text().strip() + ") "
                    chapter_text += (paragraph_element.get_text().strip() + " ")
            sentences.append(chapter_text)

def clean_string(unclean_string): 
    unclean_string = unclean_string.replace("*", "")
    iterable_matches = re.finditer('\[\w+\]', unclean_string)
    square_spans = []
    for match in iterable_matches:
        square_spans.append(match.span())
    clean_string = ""
    start = 0
    for span in square_spans:
        clean_string += unclean_string[start:span[0] - 2]
        start = span[1]
    clean_string += unclean_string[square_spans[-1][1]:]
    return clean_string

with open(DATA_DIR+"english-chapters.txt", "w+", encoding="utf-8") as english_file:
    unclean_string = "\n".join(sentences)
    clean_string = clean_string(unclean_string)
    english_file.write(clean_string)