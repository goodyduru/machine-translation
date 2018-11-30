from bs4 import BeautifulSoup
import re

toc_file = "./data/eng/OEBPS/1001061103.xhtml"
books = []
sentences = []
html_files = []

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
    sentences.append(title)
    #Get Old Testament and New Testament headers
    headers = soup.find_all("h2")
    for header in headers:
        sentences.append(replace_some_symbols_with_spaces(header.string.lower().capitalize()))
    #Get Content Headers
    for i in [3, 5, 6]:
        content_header = soup.find(attrs={"data-pid": i})
        sentences.append(content_header.string.lower().capitalize())
    #Get Bible Chapters For Old Testament
    for i in range(7, 160, 4):
        book = soup.find(attrs={"data-pid": i})
        books.append(replace_some_symbols_with_spaces(book.string))
        html_files.append(book.a['href'])
    #Get Bible Chapters For New Testament
    for i in range(168, 273, 4):
        book = soup.find(attrs={"data-pid": i})
        books.append(replace_some_symbols_with_spaces(book.string))
        html_files.append(book.a['href'])