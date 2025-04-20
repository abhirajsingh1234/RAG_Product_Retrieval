from ast import literal_eval
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pinecone import Pinecone
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "rag"

description = pc.describe_index(name=index_name)

index = pc.Index(host=description.host)

model = OllamaEmbeddings(model = 'nomic-embed-text:latest')

parser= JsonOutputParser()

df = pd.read_csv("data.csv")

df['embeddings'] = df['embeddings'].apply(literal_eval)

# print(df.iloc[0])

question ='what is the price of the asus vivobook'

query_embedding = model.embed_query(question)

data = index.query(vector=query_embedding, top_k=2, include_values=True)

content = []

for i in range(len(data['matches'])):

    id = data['matches'][i]['id']

    row = df.iloc[int(id)]

    content.append(row['content'])

response_model = GoogleGenerativeAI(model='gemini-1.5-pro',api_key=os.getenv("GEMINI_API_KEY"))

template = PromptTemplate(

    input_variables=["question","context"],

    partial_variables={"format": parser.get_format_instructions()},

        template  = """you are a helpful assistant.

        given the question and the unstructured context generate the answer.

        answer should resolve user query.

        format:{format}

        question: {question}

        context: {context}
        """
)
chain = template | response_model | parser

result = chain.invoke({"question": question, "context": content})

print(result)








