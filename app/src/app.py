# external modules


from flask import Flask, request, render_template, redirect, session, url_for, flash
from flask_session import Session

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

import subprocess
import os
import shutil
import tempfile

# internal modules
from rag import PolicyRAG

### Main flask application ###
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
STATIC_DIR = os.path.join(DIR_PATH, "../static")
TEMPLATE_DIR = os.path.join(DIR_PATH, "../templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config["SECRET_KEY"] = "randomkey"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"]= False
app.config.update(SESSION_COOKIE_SAMESITE="None", SESSION_COOKIE_SECURE=True)

Session(app)

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
    uploaded_files = list(session["storage"]["user-server-names"])

  return render_template("index.html", **locals())

@app.route("/upload", methods=["POST"])
async def upload() -> str:
  """
    @params:  N/A
    @purpose: upload a file in request.files to session["storage"] TemporaryDirectory object
    @returns: str - HTML presented to user
  """
  if "storage" not in session:
    session["storage"] = {"handle": tempfile.TemporaryDirectory(delete=False)}
    session["storage"]["path"] = session["storage"]["handle"].name
    session["storage"]["user-server-names"] = {}

  user_server_names = {}
  for file in request.files.getlist("files"):
    if file.filename != '' and file.filename not in session["storage"]["user-server-names"]:
      with tempfile.NamedTemporaryFile(dir=session["storage"]["path"], delete=False, suffix=".pdf") as tfile:
        tfile.write(file.read())
        user_server_names.update({file.filename: tfile.name})
    else:
      print("File is null or file.filename is in session")

  updated_storage = session.get("storage")
  updated_storage["user-server-names"] |= user_server_names
  session["storage"] = updated_storage

  print(session["storage"])

  return redirect(url_for("root"))

@app.route("/submit", methods=["POST"])
async def submit() -> str:
  """
    @params:  N/A
    @purpose: submit a prompt to the LLM
    @returns: str - HTML presented to user
  """
  if "history" not in session:
    session["history"] = []

  prompt = request.form["prompt"]
  if prompt is not None:
    session["history"] = session.get("history") + ["question: " + prompt]
    
    if session.get("storage"):
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
      docs = PyPDFDirectoryLoader(path=session["storage"]["path"]).load()

      if len(docs) == 0:
        flash(f"No PDF files found")
        subprocess.run(["ls", "-al", session["storage"]["path"]])
        return redirect(request.referrer)

      chunks = text_splitter.split_documents(docs)
      chunks = filter_complex_metadata(chunks)

      vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
      retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
          "k": 4,
          "score_threshold": 0.4,
        },
      )
    else:
      print("No storage in session...")
      retriever = None

    print(retriever)

    llm = ChatOllama(base_url="http://ollama:11434", model="mistral")
    #llm = OpenAI()
    rag = PolicyRAG(llm, retriever) # eventually chunk and cache files, check if files changed to rerun embeddings
    
    answer = rag.run(prompt)

    session["history"] = session.get("history") + ["answer: " + answer]

  return redirect(request.referrer)

@app.route("/clear-history")
async def clear_history():
  """
    @params:  N/A
    @purpose: pop "history" from session
    @returns: str - HTML presented to user
  """
  session.pop("history", None)
  flash("Conversation history successfully cleaned!")
  return redirect(request.referrer)

@app.route("/clear-storage")
async def clear_storage():
  """
    @params:  N/A
    @purpose: cleanup temporary directory and pop "storage" from session
    @returns: str - HTML presented to user
  """
  if "storage" in session:
    session["storage"]["handle"].cleanup()
  session.pop("storage", None)
  flash("Session storage successfully cleaned!")
  return redirect(request.referrer)

if __name__ == "__main__":
  # run with debug=True for hot reloading
  app.run(debug=True)