from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from flask import Flask, request
app = Flask(__name__)

@app.route("/")
def root():
  return "<h1>Booyah flask and docker!</h2>"

@app.route("/ollama", methods=("get", "post"))
def ollama():
  topic = request.args.get("topic")
  if topic is not None:
    llm = Ollama(base_url="http://ollama:11434", model="mistral")
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"topic": topic})
    return f"Topic: {topic}<br>Joke: {answer}"
  else:
    return r"""
<h1> Enter topic </h1>
<form action="/ollama">
  <input name="topic" value="Bears" />
  <input type="submit" /> 
</form>
"""

if __name__ == "__main__":
  app.run(debug=True)