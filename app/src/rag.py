# external modules
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import TextLoader

from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models import BaseChatModel

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from qdrant_client import QdrantClient

from typing import Optional

# internal modules
# N/A

class PolicyRAG():
  model: Optional[BaseChatModel] = None
  doc_retriever: Optional[VectorStoreRetriever] = None
  policy_retriever: Optional[VectorStoreRetriever] = None
  chain: Runnable = None

  def __init__(
    self, 
    model: BaseChatModel,
    retriever: Optional[VectorStoreRetriever] = None
  ) -> None:
    """
      @params:  model - BaseChatModel used in LCEL self.chain, for now almost always a ChatOllama instance,
                retriever - VectorStoreRetriever used in addition to VectorStoreRetriever for Qdrant policies
      @purpose: construct a RAG wrapper around `model` that uses `retriever` and 
                a Qdrant database of policies as context.
      @returns: N/A
    """

    url = "http://qdrant_policies:6333" #todo make this env
    embeddings = FastEmbedEmbeddings()
    client = QdrantClient(url=url)
    qdrant = Qdrant(
      client=client,
      collection_name="ubc_pdf_policies", #todo make this env
      embeddings=embeddings,
    )
    policy_retriever = qdrant.as_retriever(
      search_type="similarity_score_threshold",
      search_kwargs={
        "k": 4,
        "score_threshold": 0.4,
      },
    )

    self.model = model
    self.context = { 
      "question": RunnablePassthrough(),
      "policy_context": policy_retriever,
    }

    if retriever is not None:
      self.context["document_context"] = retriever
      self.prompt = PromptTemplate.from_template(r"""
          You are an assistant for question-answering tasks. Use the following pieces of retrieved 
          context from policy and uploaded documents to answer the question. If you don't know the answer, just
          say that you don't know. Use three sentences maximum and keep the answer concise.

          Question: {question}
          Policy Context: {policy_context}
          Document Context: {document_context}
          Helpful Answer: 
        """)
    else:
      self.prompt = PromptTemplate.from_template(r"""
          You are an assistant for question-answering tasks. Use the following pieces of retrieved 
          context from policy and uploaded documents to answer the question. If you don't know the answer, just
          say that you don't know. Use three sentences maximum and keep the answer concise.

          Question: {question}
          Policy Context: {policy_context}
          Helpful Answer: 
        """)

    self.chain = (self.context
      | self.prompt
      | self.model
      | StrOutputParser())

  def run(self, query: str) -> str:
    return self.chain.invoke(query)

"""
# without RAG:
  prompt = ChatPromptTemplate.from_template("Tell me a short, 1 sentence joke about the topic: {topic}")
  llm = ChatOllama(base_url="http://ollama:11434", model="mistral")
  # construct chain using prompt and ollama instance running on other container
  chain = prompt | llm | StrOutputParser()

  answer = chain.invoke({"topic": topic})
# with RAG:
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
        \"""
        <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
        to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
          maximum and keep the answer concise. [/INST] </s> 
        [INST] Question: {question} 
        Context: {context} 
        Answer: [/INST]
        \"""
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
    print(rag.query("What were some names of prevalent behaviourists and what did they believe?"))

  if __name__ == "__main__":
    print("Running app.py!")
    main()
"""