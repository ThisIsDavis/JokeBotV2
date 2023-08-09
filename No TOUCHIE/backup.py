import gradio as gr
import os
from llama_index import SimpleDirectoryReader
from llama_index import GPTSimpleVectorIndex, ServiceContext
from llama_index.readers.file.docs_parser import PDFParser
# from llama_index.readers.schema.base import Document
from llama_index.readers.schema.base import Document


class GPTProcessing(object):
    
    def __init__(self, ui_obj):
        self.name = "Custom Data Processing with ChatGPT"
        self.description = "Add Custom PDF/txt Data Processing with ChatGPT"
        self.api_key = None
        self.index_folder = "indexData" #remember to put output index folder name
        self.ui_obj = ui_obj
        self.api_key_status = "Error: OpenAI API is not set"
        self.selected_index = None
        self.index_status = "Error: Index is not selected"
        self.index_setup_result = ".... no action..."

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown("Custom Fine-tuning with Large Language Model")
            #now we set up the tabs in ui
            with gr.Tabs():
                #first tab item
                with gr.TabItem("Setup OpenAI Configuration"):
                    with gr.Row():   #top row
                        openai_api_key = gr.Textbox(label="OpenAI API Key",
                                                    placeholder="OpenAI API key here..",
                                                    type='password')
                        #like a button click listener
                        set_api_action = gr.Button("Setup API Key")
                    with gr.Row():   #second row
                        #to show openAi status 
                        openai_api_stats = gr.Label(self.api_key_status)
                #second tab item
                with gr.TabItem("Training/Fine-tuning with Custom Data"):
                    with gr.Row():
                        source_data = gr.File(
                            label="Single PDF or Text files only",
                            file_count="single",
                            file_types=["file"])
                        index_setup_action = gr.Button("Create Index")
                    with gr.Row():
                        source_data = gr.Files(
                            label="Multiple Text files only",
                            file_count="multiple",
                            file_types=["file"])
                        index_multiple_setup_action = gr.Button("Create Index for Multiple Files")
                    with gr.Row():
                        index_setup_result_label = gr.Label(self.index_setup_result)
                #third tab item
                with gr.TabItem("Query Custom Data"):
                    with gr.Row():
                        index_listing_action = gr.Button("List all Indexes")
                        index_list_output = gr.Textbox(label="Listed Indexes")
                    with gr.Row():
                        index_file_name = gr.Textbox(label="Select Index")
                        index_selection_action = gr.Button("Load Index")
                    with gr.Row():
                        index_status_label = gr.Label(self.index_status)
                    with gr.Row():
                        #this part accepts queries
                        query_question = gr.Textbox(label="Enter your question:", lines=5)
                        #button to get answer
                        query_data_action = gr.Button("Get Answer")
                    with gr.Row():
                        #to show response
                        query_result_text = gr.Textbox(label="Query Result:", lines=10)
            
            #setup first click button
            set_api_action.click(
                self.update_api_status,
                [
                    openai_api_key
                ],
                [
                    openai_api_stats
                ]
            )

            #setup second click button
            index_setup_action.click(
                self.index_setup_process,
                [
                    source_data    #input
                ],
                [
                    index_setup_result_label  #output
                ]
            )

            #setup second create index button
            index_multiple_setup_action.click(
                self.index_setup_process,
                [
                    source_data    #input
                ],
                [
                    index_setup_result_label  #output
                ]
            )

            index_listing_action.click(
                self.index_listing,
                [

                ],
                [
                    index_list_output
                ]
            )

            index_selection_action.click(
                self.setup_index_from_collection,
                [
                    index_file_name
                ],
                [
                    index_status_label
                ]
            )

            query_data_action.click(
                self.get_answer_from_index,
                [
                    query_question
                ],
                [
                    query_result_text
                ]
            )

    ############################## Helper Functions#################################################################
    def launch_ui(self):
        self.ui_obj.launch(share=True)


    #to update api key
    def update_api_status(self, api_key):
        if api_key is not None and len(api_key) > 0:
            self.api_key = str(api_key)
            self.api_key_status = str("Success: OpenAI key is set")
            os.environ["OPENAI_API_KEY"] = self.api_key
        return self.api_key_status
    
    #method for second button
    def load_source_data(self, file):
        source_data = []
        data_list = []
        file = file.lower()
        file_extension = os.path.splitext(file)
        if len(file_extension) == 2 and file_extension[1] in ['.pdf', '.txt']:
            if file_extension[1] == '.pdf':
                parser = PDFParser()
                extracted_pdf = parser.parse_file(file)
                data_list.append(extracted_pdf)
                source_data = [Document(d) for d in data_list]
            elif file_extension[1] == '.txt':
                text_file_path = os.path.split(file)
                source_data = SimpleDirectoryReader(text_file_path[0]).load_data()
        return source_data
    
    # #load multiple txt files (for project use)
    # def load_source_multiple_data(self, path):
    #     source_data_multiple = SimpleDirectoryReader(path).load_data()

    #     return source_data_multiple

    
    #to index the file
    def index_setup_process(self, file):
        source_documents = self.load_source_data(file.name)
        # multiple_source_documents = self.load_source_multiple_data(file.name)
        status_message = "Error: Unable to create to source document index"
        if len(source_documents) > 0:
            source_index = GPTSimpleVectorIndex.from_documents(source_documents)  #get the index from gpt api
            saved_file = self.save_index_document(source_index, file.name) #save the source index
            if saved_file is not None:
                status_message = "Success: The index is ready as [" + saved_file + "]" #this will be shown in the label row
        return status_message
    

    #to save the vector index. .json in indexData folder.
    def save_index_document(self, source_index, out_file_name):
        try:
            final_out_file = os.path.basename(out_file_name.lower()) + ".json"
            final_out_file_path = os.path.join(os.getcwd(), self.index_folder, final_out_file)
            source_index.save_to_disk(final_out_file_path)
        except:
            final_out_file_path = None

        return final_out_file_path  #will see index.json in directory.
    

    #list the indexes created. (nothing to do with API methods.)
    def index_listing(self):
        index_path = os.path.join(os.getcwd(), self.index_folder) #accesses the indexData folder
        all_files = os.listdir(index_path)
        if len(all_files) == 0:
            return "No files"
        else:
            return all_files
        
    def setup_index_from_collection(self, index_name):
        if index_name is not None and len(index_name) > 0 and self.api_key is not None:
            index_path = os.path.join(os.getcwd(), self.index_folder, index_name)
            if os.path.exists(index_path):
                self.selected_index = GPTSimpleVectorIndex.load_from_disk(index_path)
                self.index_status = "Success: Index is set.."
            else:
                self.index_status = "Error: Index path is bad"
        return self.index_status

    def get_answer_from_index(self, query_question):
        query_result = "Error: Unable to get answer to your question"
        if query_question is not None and len(query_question) > 0:
            if self.selected_index:
                query_result = self.selected_index.query(query_question)
        return query_result

    

if __name__ == '__main__':
    my_app = gr.Blocks()
    #this will call our program with gr.Block from gradio
    gradio_ui = GPTProcessing(my_app)
    gradio_ui.create_ui()
    #to launch the ui
    gradio_ui.launch_ui()


