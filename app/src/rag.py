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
          context from policy documents to answer the question. If you don't know the answer, just
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