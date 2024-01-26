
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms.fireworks import Fireworks
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class RAGModel:
  vector_store = None
  retriever = None
  chain = None

  def __init__(self, model):
    self.model = model
    self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    self.prompt = PromptTemplate.from_template(
      """
      <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
      to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
        maximum and keep the answer concise. [/INST] </s> 
      [INST] Question: {question} 
      Context: {context} 
      Answer: [/INST]
      """
    )

  def setup(self, pdf_source_dir: str):
    docs = PyPDFDirectoryLoader(path=pdf_source_dir).load()

    if len(docs) == 0:
      raise FileNotFoundError(f"no pdf files found in {pdf_source_dir}")

    chunks = self.text_splitter.split_documents(docs)
    chunks = filter_complex_metadata(chunks)

    vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
    self.retriever = vector_store.as_retriever(
      search_type="similarity_score_threshold",
      search_kwargs={
        "k": 3,
        "score_threshold": 0.5,
      },
    )

    self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
      | self.prompt
      | self.model
      | StrOutputParser())

  def query(self, query: str):
    if not self.chain:
      raise FileNotFoundError(f"no pdf files found")
    return self.chain.invoke(query)

def RAGModelFireworks():
  llm = Fireworks(
    model="accounts/fireworks/models/mistral-7b-instruct-4k",
    model_kwargs={
      "temperature": 0.15,
      "max_tokens": 200,
      "top_p": 1.0
    },
  )
  return RAGModel(llm)

def main():
  rag = RAGModelFireworks()
  rag.setup("./resources/")
  print(rag.query("What were some names of prevalent Behaviourists and what did they believe?"))

if __name__ == "__main__":
  print("Running app.py!")
  main()