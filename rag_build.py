import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pinecone import ServerlessSpec, CloudProvider, AwsRegion, Metric
from pinecone import Pinecone
import pandas as pd
import time


def extract_data(folder_path):

    data = []

    for file in os.listdir(folder_path):

        if file.lower().endswith(".pdf"):

            file_path = os.path.join(folder_path, file)

            loader = PyPDFLoader(file_path)

            documents = loader.load()

            i=0

            doc_len = len(documents)

            for document in documents:

                i+=1

                data.append({"content": document.page_content, "metadata": {"source": document.metadata['source'], "total_page": doc_len, "current_page": i}})

    return data


def embedding_creator(data):

    for i in data:

        result = model.embed_query(i['content'])

        i['embedding'] = result

    return data

def is_fresh(index):

    stats = index.describe_index_stats()

    vector_count = stats.total_vector_count

    print(f"Vector count: ", vector_count)

    return vector_count > 0

def data_embed_saver(emded_data):

    indexes =[]

    content=[]

    embeddings=[]

    metadata=[]

    for ind,data in enumerate(emded_data):

        indexes.append(str(ind))

        content.append(data['content'])

        embeddings.append(data['embedding'])

        metadata.append(data['metadata'])

    df = pd.DataFrame({"indexes": indexes, "content": content, "embeddings": embeddings, "metadata": metadata})

    index.upsert(vectors=zip(df.indexes, df.embeddings, df.metadata))

    df['embeddings'] = df['embeddings'].apply(str)

    df.to_csv("data.csv", index=False, escapechar='\\')
    
    
if __name__ == '__main__':

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "rag"

    # Delete the index if an index of the same name already exists

    if pc.has_index(name=index_name):

        pc.delete_index(name=index_name)

    pc.create_index(

        name=index_name,

        metric=Metric.COSINE,

        dimension=768,

        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1),

    )

    description = pc.describe_index(name=index_name)

    index = pc.Index(host=description.host)

    folder_path = "pdf_data"

    model = OllamaEmbeddings(model = 'nomic-embed-text:latest')

    content = extract_data(folder_path)

    emb_content = embedding_creator(content)

    data_embed_saver(emb_content)

    while not is_fresh(index):
        # It takes a few moments for vectors we just upserted
        # to become available for querying
        time.sleep(5)

