import os

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

reader = PdfReader('sample.pdf')

text = ''

for page in reader.pages:
    text += page.extract_text()

with open('output.txt', 'w') as file:
    file.write(text)

loader = DirectoryLoader(
    './',
    glob='**/*.txt',
    loader_cls=TextLoader
)

documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1024,
    chunk_overlap=128
)

texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

persist_directory = 'db'

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)

vectordb.persist()
