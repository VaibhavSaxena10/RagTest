from langchain_community import vectorstores
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

model = ChatGroq(model = "qwen/qwen3.8-27b")

loader = PyPDFLoader("cricket_notes.pdf")

documents = loader.load()
#print(documents[0])

Spliter = RecursiveCharacterTextSplitter( chunk_size = 400,
chunk_overlap = 0)

result = Spliter.split_documents(documents)

embedding = HuggingFaceEmbeddings( model_name= "sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(result,embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt = ChatPromptTemplate.from_template("""Answer the question based on the context provided and if you don't know anything about the question asked simply return "I don't Know about that ":
context: {context}
question: {question}
""")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

print(chain.invoke({"context": retriever.invoke("Wickets"), "question": "Lenght of the wickets"}))