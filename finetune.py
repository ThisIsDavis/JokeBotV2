from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex
import os


#Open AI Key
os.environ["OPENAI_API_KEY"] = "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"

#read files in directory
loaded_content = SimpleDirectoryReader('sourceData').load_data()

#this will generate ouput index (this will require an openAI key)
output_index = GPTVectorStoreIndex(loaded_content)
output_index.save_to_disk("indexData/index.json")

