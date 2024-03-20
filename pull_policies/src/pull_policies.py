
# external modules
import requests

from bs4 import BeautifulSoup

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models

import re
import os
import datetime

# internal modules
# N/A

URL = "https://universitycounsel.ubc.ca/policies/"
URL_ROOT = "https://universitycounsel.ubc.ca"
REGEX_URL = r"^https://universitycounsel\.ubc\.ca/policies/.+$"
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
      if pdf.match(href):
        files.append(href)
  return files

def save_files(file_paths):
  for path in file_paths:
    response = requests.get(URL_ROOT + path)
    file_name = Path(path).name
    file_path = DESTINATION + file_name
    with open(file_path, "wb") as file:
      print(f"Saving {file_name} to {file_path}...")
      file.write(response.content)

def embed_files_into_qdrant():
  FAST_EMBED_EMBEDDINGS = "BAAI/bge-small-en-v1.5"
  COLLECTION_NAME = "ubc_pdf_policies"
  # localhost no work, need host.docker.internal
  # see: https://github.com/qdrant/qdrant-client/issues/105#issuecomment-1423214105 
  qdrant_client = QdrantClient(host="host.docker.internal", port=6333)
  qdrant_client.set_model(FAST_EMBED_EMBEDDINGS)
  # delete old embeddings
  try:
    qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
  except:
    print(f"collection {COLLECTION_NAME} doesnt exist")
  # generate new collection
  qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
  )
  # load and chunk documents
  # TODO
  # generate embeddings from chunks
  # TODO
  # upsert embeddings into database
  collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
  # qdrant_client.upsert(
  #   collection_name=COLLECTION_NAME,
  #   ...
  # )
  print(collection_info)

def main():
  load_files = False
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
  print("DONE")
  print(datetime.datetime.today())
