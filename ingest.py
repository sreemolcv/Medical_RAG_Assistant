from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ fixed import
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
import os

all_docs = []

pdf_folder = "data/papers"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        path = os.path.join(pdf_folder,file)
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)
print(f"Loaded {len(all_docs)} pages")
#print(docs[0].page_content)
##### Chunk Documents ######

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

print(len(chunks)) 

#### Create Embeddings ###
embeddings = HuggingFaceEmbeddings(
  model_name = 'BAAI/bge-small-en-v1.5'
 )

# Store in ChromaDB

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./db"
)
print("✅ Ingestion complete. Chunks stored:", len(chunks))

# Query the document

db = Chroma(
    persist_directory= "./db",
    embedding_function=embeddings
)

#Retrieve Documents

retriever = db.as_retriever(
    search_kwargs={"k":5}
)

llm = ChatOllama(
    model="llama3"
)

def rag_chain(question):

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Answer only from the context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content


# for doc in docs:
#     print(
#         doc.metadata["source"],
#         doc.metadata["page"]
#     )