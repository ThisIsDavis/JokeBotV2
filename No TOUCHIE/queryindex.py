from llama_index import GPTSimpleVectorIndex
import os

#Open AI Key
os.environ["OPENAI_API_KEY"] = "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"


loaded_index = GPTSimpleVectorIndex.load_from_disk("indexData/index.json")


prompt = "Ask me"

while prompt != 'exit':
    prompt = input("Prompt > ")
    res = loaded_index.query(prompt)
    print(res)