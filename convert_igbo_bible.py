import os
import re
from bs4 import BeautifulSoup

DIR = "./data/igbo/OEBPS/"
DATA_DIR = "./data/"
toc_file = DIR + "1001060401.xhtml"
books = []
headers = []
chapter_one_files = []
html_files = {}
sentences = []
"""
    This processes the igbo table of content file
"""
with open(toc_file, encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")
    #Get Title and remove colon
    title = soup.title.string[:-1]
    headers.append(title)
    #Get Old Testament and New Testament headers
    h2_headers = soup.find_all("h2")
    for header in h2_headers:
        headers.append(header.string)
    #Get Content Headers
    for i in range(3, 6):
        content_header = soup.find(attrs={"data-pid": i})
        headers.append(content_header.string.lower().capitalize())
    #Get Bible Chapters For Old Testament
    for i in range(6, 121, 3):
        book = soup.find(attrs={"data-pid": i})
        books.append(book.string)
        chapter_one_files.append(book.a['href'].split(".")[0])
    #Get Bible Chapters For New Testament
    for i in range(124, 203, 3):
        book = soup.find(attrs={"data-pid": i})
        books.append(book.string)
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
            paragraph_elements = soup.find_all(attrs={"class": re.compile("p\d+")})
            chapter_text = ""
            
            for paragraph_element in paragraph_elements:
                classes = paragraph_element['class']

                if len(classes) > 0:
                    chapter_text += paragraph_element.get_text()
            sentences.append(chapter_text)


""" This is going to find the bracketed words usually found in 
    Psalms e.g [Aleph] and remove them along with the hebrew character 
    found before them. Also it's going to remove the * found in the chapters.
    Also remove other bracketed words.
"""
def clean_string(unclean_string): 
    unclean_string = unclean_string.replace("*", "")
    iterable_matches = re.finditer('\[\w+\]', unclean_string)
    square_spans = []
    for match in iterable_matches:
        square_spans.append(match.span())
    no_hebrew_string = ""
    start = 0
    for span in square_spans:
        no_hebrew_string += unclean_string[start:span[0] - 2]
        start = span[1]
    no_hebrew_string += unclean_string[square_spans[-1][1]:]
    clean_string = re.sub('\[.*\]', "", no_hebrew_string)
    return clean_string
    

with open(DATA_DIR+"igbo.txt", "w+", encoding="utf-8") as igbo_file:
    unclean_string = "\n".join(sentences)
    clean_string = clean_string(unclean_string)
    igbo_file.write(clean_string)
