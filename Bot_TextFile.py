from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(model = "qwen/qwen3.8-27b")


loader = TextLoader("cricket_info.txt", encoding="utf-8")

doc = loader.load()

#print(doc[0])

splitter = RecursiveCharacterTextSplitter(chunk_size = 400 , chunk_overlap = 20)

result = splitter.split_documents(doc)

print(len(result))
print(result[0])

embedding = HuggingFaceEmbeddings ( model_name = "BAAI/bge-large-en")

vectorstore = FAISS.from_documents(doc,embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt = PromptTemplate(
    template = "Get me the answer based on the context provided {context} and answer the question {question} and if you don't have the answer then say I don't know about that ",
    input_variables=["context", "question"]
)
parser = StrOutputParser()
chain = prompt | model | parser

print(chain.invoke({"context":retriever.invoke("IPL"), "question":"What does ipl stands for?"}))