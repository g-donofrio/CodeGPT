import os

from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.llms import OpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.chains.question_answering import load_qa_chain

class SingleModelCode:

    openai_api_key = None
    repo_url = None
    repo_path = None
    suffixes = None
    language = None
    documents = None
    db = None
    retriver = None
    llm = None
    memory = None
    qa = None

    chat_history = []

    def __init__(self, openai_api_key = None, repo_url = None, repo_path = None, repo_dest=None, db_directory=None, suffixes=[".py"], language=Language.PYTHON):

        if not os.path.isdir(repo_dest):
            # Clone repo from Github
            os.makedirs(repo_dest)
            print("DEBUG: Clone the repo")
            Repo.clone_from(repo_url, to_path=repo_dest)
        else:
            print("DEBUG: Local repo")
        # Load
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*",
            suffixes=suffixes,
            parser=LanguageParser(language=language, parser_threshold=500)
        )
        print("DEBUG: Load files")
        documents = loader.load()
        python_splitter = RecursiveCharacterTextSplitter.from_language(language=language, 
                                                               chunk_size=2000, 
                                                               chunk_overlap=200)
        print("DEBUG: Split documents")
        texts = python_splitter.split_documents(documents)
        db = None
        if not os.path.isdir(db_directory):
            print("DEBUG: Create Chroma DB")
            db = Chroma.from_documents(texts, OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=()), persist_directory=db_directory)
            db.persist()
        else:
            print("DEBUG: Load Chroma DB")
            db = Chroma(persist_directory=db_directory, embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=()))
        print("DEBUG: Create retriver")
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8},
        )
        print("DEBUG: Instanciate LLM")
        llm = OpenAI(openai_api_key = openai_api_key, temperature=0.1, verbose=True)
        # print("DEBUG: Create memory")
        # memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
        # qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=True)
        print("DEBUG: Instanciate QA model")
        qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)

        print("DEBUG: Engine ready.")

        # Save properties

        self.openai_api_key = openai_api_key
        self.repo_url = repo_url
        self.repo_path = repo_path
        self.repo_dest = repo_dest
        self.db_directory = db_directory
        self.suffixes = suffixes
        self.language = language

        self.documents = documents
        self.db = db
        self.retriver = retriever
        self.llm = llm
        # self.memory = memory
        self.qa = qa
    
    """ 
        Output= {
            "answer",
            "sources" 
        }

        sources: [{
            "content",
            "metadata"
        }]

    """
    def __call__(self, prompt):
        result = self.qa({"question": prompt, "chat_history": self.chat_history})
        tmp_sources = []
        for source in result["source_documents"]:
            tmp_sources.append({"content": source.page_content, "metadata": source.metadata})
        return {
            "answer": result["answer"],
            "sources": tmp_sources
        }