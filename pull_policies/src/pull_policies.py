
# external modules
import requests

from bs4 import BeautifulSoup

from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFDirectoryLoader

import re
import os
import datetime

# internal modules
# N/A

# policy path constant
URL = "https://universitycounsel.ubc.ca/policies/"
# universitycounsel website root constant
URL_ROOT = "https://universitycounsel.ubc.ca"
# regex for each policy pdf href looks like for the anchor tags
REGEX_URL = r"^https://universitycounsel\.ubc\.ca/policies/.+$"
# volume for storing policies
DESTINATION = "/root/.qdrant_policies/policies/"

def determine_policy_locations():
  policy = re.compile(REGEX_URL)
  page = requests.get(URL)
  soup = BeautifulSoup(page.content, "html.parser")
  anchors = soup.find_all("a", href=True)
  return [a["href"] for a in anchors if policy.match(a["href"])]

def find_files_from_locations(locs):
  pdf = re.compile(r'.+\.pdf$')
  files = []
  for loc in locs:
    page = requests.get(loc)
    soup = BeautifulSoup(page.content, "html.parser")
    anchors = soup.find_all("a", href=True)
    for anchor in anchors:
      href = anchor["href"]
      # is the href going to a pdf?
      if pdf.match(href):
        # stick it in the file list!
        files.append(href)
  return files

def save_files(file_paths):
  for path in file_paths:
    try:
      # save file given pdf resource path
      response = requests.get(URL_ROOT + path)
      file_name = Path(path).name
      file_path = DESTINATION + file_name
      with open(file_path, "wb") as file:
        print(f"Saving {file_name} to {file_path}...")
        file.write(response.content)
    except:
      print(f"Unable to save file from {URL_ROOT + path}")

def embed_files_into_qdrant():
  embeddings = FastEmbedEmbeddings()
  
  # split text documents stored in DESTINATION
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
  loader = PyPDFDirectoryLoader(path=DESTINATION)
  documents = loader.load()
  texts = text_splitter.split_documents(documents)

  # create embeddings from policies, Qdrant.from_documents truncates and recreates
  # collection with collection_name
  COLLECTION_NAME = "ubc_pdf_policies" #todo make this env
  URL = "http://qdrant_policies:6333" #todo make this env
  qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=URL,
    prefer_grpc=False,
    collection_name=COLLECTION_NAME,
  )

def main():
  load_files = True
  if load_files:
    locs = determine_policy_locations()
    print("Locations determined...")
    files = find_files_from_locations(locs)
    print("Files taken from locations...")
    save_files(files)
    print("Files saved...")
  embed_files_into_qdrant()
  print("Files embedded into qdrant...")

if __name__ == "__main__":
  main()
  print("DONE PULLING AND EMBEDDING POLICIES")
  print(datetime.datetime.today())
  print("'app' container can now be used!")
