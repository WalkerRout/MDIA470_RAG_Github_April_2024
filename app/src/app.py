from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def root():
  """
  @params:  N/A
  @purpose: present root template
  @returns: str - HTML presented to user
  """

  return r"""
    <a href="/ollama"><h1>Ollama Model</h2></a>
  """

@app.route("/ollama", methods=("get", "post"))
def ollama():
  """
  @params:  N/A
  @purpose: accept query string with argument `topic`, feed topic into langchain chain
            using Ollama as a model
  @returns: str - HTML presented to user
  """

  topic = request.args.get("topic")

  if topic is not None:
    # construct chain using prompt and ollama instance running on other container
    prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    llm = ChatOllama(base_url="http://ollama:11434", model="mistral")
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
  # run with debug=True for hot reloading
  app.run(debug=True)