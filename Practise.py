import chromadb
import anthropic

with open("Knowledge.txt", "r") as f:
    text = f.read()

chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

print(f"Loaded {len(chunks)} chunks from knowledge.txt")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="company_docs")

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"]
    )
anthropic_client = anthropic.Anthorpic(api_key="")
print("\n Document Q&A ready! type 'quit' to exit. \n")

while True:
    question = input("Ask a question about your docs: ")

    if question.lower() == "quit":
        print("Goodbye")
        break

    result = collection.query(
        query_texts=[question],
        n_results=2
    )

    context = "\n".join(result["documents"][0])

    message = [
        {
            "role": "user"
            "content": f"""Aswer te question using the context only below.
            if the anseer is not inthe ccontet say I dont have that informationin my document.result

            Context:{context}

question:{question}"""
        }
    ]

response = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens = 1024,
    system="you are a helpful assistant. Answer only based on the providedcontext. be concise and precise",
    messages=message
)


print(f"\nAI: {response.content[0].text}\n")

