from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
import os


#Open AI Key
os.environ["OPENAI_API_KEY"] = "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"


#read files in directory
loaded_content = SimpleDirectoryReader('sourceData').load_data()

#this will generate ouput index (this will require an openAI key)
output_index = GPTVectorStoreIndex(loaded_content)
output_index.save_to_disk("indexData/index.json")

