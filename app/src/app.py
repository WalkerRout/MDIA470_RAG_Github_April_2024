# external modules
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from flask import Flask, request, render_template, redirect, session, url_for, flash
from flask_session import Session

import os
import tempfile

# internal modules
from rag import PolicyRAG

### Main flask application ###
dir_path = os.path.dirname(os.path.realpath(__file__))
static_dir = os.path.join(dir_path, "../static")
template_dir = os.path.join(dir_path, "../templates")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config["SECRET_KEY"] = "randomkey"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"]= False

Session(app)

@app.route("/")
async def root() -> str:
  """
    @params:  N/A
    @purpose: create local variables to serve with index.html jinja2 template
    @returns: str - HTML presented to user
  """
  history = ""
  if "history" in session:
    history = session["history"]
  return render_template("index.html", **locals())

@app.route("/upload", methods=["POST"])
async def upload() -> str:
  """
    @params:  N/A
    @purpose: upload a file in request.files to session["storage"] TemporaryDirectory object
    @returns: str - HTML presented to user
  """
  if "storage" not in session:
    session["storage"] = tempfile.TemporaryDirectory()

  for file in request.files.getlist("files"):
    if file.filename != '':
      print(file)

  return redirect(request.referrer)

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

    llm = ChatOllama(base_url="http://ollama:11434", model="mistral")
    rag = PolicyRAG(llm) # eventually chunk and cache files, check if files changed to rerun embeddings
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
    session["storage"].cleanup()
  session.pop("storage", None)
  flash("Session storage successfully cleaned!")
  return redirect(request.referrer)

if __name__ == "__main__":
  # run with debug=True for hot reloading
  app.run(debug=True)