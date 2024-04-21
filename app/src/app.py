# external modules
from flask import Flask, request, render_template, redirect, session, url_for, flash
from flask_session import Session

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient

import subprocess
import os
import shutil
import tempfile

# internal modules
from rag import PolicyRAG
from user_storage import UserStorage

### Main flask application ###
ALLOWED_EXTENSIONS = ["pdf"]

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# directory for static data (css, js, .png, etc)
STATIC_DIR = os.path.join(DIR_PATH, "../static")
# directory for templates (index.html)
TEMPLATE_DIR = os.path.join(DIR_PATH, "../templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["SECRET_KEY"] = "randomkey"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"]= False
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

Session(app)

def allowed_file(file_name: str) -> bool:
  """
    @params:  file_name {str} - name of file to verify
    @purpose: check whether a file name is valid by making sure its
              extension is under ALLOWED_EXTENSIONS
    @returns: bool - true if file is allowed, false if file is not allowed
  """
  return '.' in file_name and \
    file_name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
async def root() -> str:
  """
    @params:  N/A
    @purpose: create local variables to serve with index.html jinja2 template
    @returns: str - HTML presented to user
  """
  history = []
  if session.get("history"):
    history = session["history"]

  uploaded_files = []
  if session.get("storage"):
    uploaded_files = session["storage"].upload_names()

  return render_template("index.html", **locals())

"""
 ####### FILE UPLOAD IS NOT YET COMPLETE, THIS IS WHAT IT MAY LOOK LIKE #######
@app.route("/upload", methods=["POST"])
async def upload() -> str:
  # @params:  N/A
  # @purpose: upload a file in request.files to session["storage"] TemporaryDirectory object
  # @returns: str - HTML presented to user
  if "storage" not in session:
    session["storage"] = UserStorage()

  storage = session["storage"]
  for file in request.files.getlist("files"):
    if file.filename != '' \
      and file.filename not in storage.upload_names() \
      and allowed_file(file.filename):
      storage.add_file(file.filename, file.read())
    else:
      print("File is null or file.filename is invalid (in session or not supported)")
  session["storage"] = storage

  return redirect(url_for("root"))
"""

@app.route("/submit", methods=["POST"])
async def submit() -> str:
  """
    @params:  N/A
    @purpose: submit a prompt to the LLM
    @returns: str - HTML presented to user
  """
  if "history" not in session:
    session["history"] = []

  answer = "Invalid prompt, please try again!"

  embeddings = FastEmbedEmbeddings()

  prompt = request.form["prompt"]
  if prompt is not None:
    session["history"] = session.get("history") + ["question: " + prompt]
    
    if session.get("storage") and len(session["storage"].upload_names()) > 0:
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
      docs = PyPDFDirectoryLoader(path=session["storage"].path()).load()

      if len(docs) == 0:
        flash(f"No PDF files found")
        print("Files:")
        subprocess.run(["ls", "-al", session["storage"].path()])
        session["storage"].cleanup()
        return redirect(request.referrer)

      chunks = text_splitter.split_documents(docs)
      chunks = filter_complex_metadata(chunks)

      vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
      retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
          "k": 4,
          "score_threshold": 0.4,
        },
      )
    else:
      flash("No Uploaded Documents")
      retriever = None

    print(retriever)

    # comment out ChatOllama line and use llm = OpenAI(api_key="your-api-key-here") to use non-local model
    llm = ChatOllama(base_url="http://ollama:11434", model="mistral")
    rag = PolicyRAG(llm, retriever) # eventually chunk and cache files, check if files changed to rerun embeddings
    
    # evaluate the answer
    answer = rag.run(prompt)

    # update history with next answer
    session["history"] = session.get("history") + ["answer: " + answer]

  # return answer and history
  return {"answer": answer, "history": session["history"]}

if __name__ == "__main__":
  # run with debug=True for hot reloading, remove when using release WSGI server
  app.run(debug=True)
