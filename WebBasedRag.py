from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")

url = "https://en.wikipedia.org/wiki/Biodiversity"
loader = WebBaseLoader(url)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
result = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    show_progress=False
)

vector_store = FAISS.from_documents(result, embedding)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate(
    template="""Answer the question based on the following context:
Context: {context}

Question: {question}

If you don't know the answer simply say: I don't know about that.""",
    input_variables=["context", "question"]
)

parser = StrOutputParser()

chain = prompt | model | parser

question = "What are animals?"
context_docs = retriever.invoke(question)

print(chain.invoke({"context": context_docs, "question": question}))
