from bs4 import BeautifulSoup

toc_file = "./data/igbo/OEBPS/1001060401.xhtml"
books = []
sentences = []
html_files = []
"""
    This processes the igbo table of content file
"""
with open(toc_file, encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")
    #Get Title and remove colon
    title = soup.title.string[:-1]
    sentences.append(title)
    #Get Old Testament and New Testament headers
    headers = soup.find_all("h2")
    for header in headers:
        sentences.append(header.string)
    #Get Content Headers
    for i in range(3, 6):
        content_header = soup.find(attrs={"data-pid": i})
        sentences.append(content_header.string.lower().capitalize())
    #Get Bible Chapters For Old Testament
    for i in range(6, 121, 3):
        book = soup.find(attrs={"data-pid": i})
        books.append(book.string)
        html_files.append(book.a['href'])
    #Get Bible Chapters For New Testament
    for i in range(124, 203, 3):
        book = soup.find(attrs={"data-pid": i})
        books.append(book.string)
        html_files.append(book.a['href'])