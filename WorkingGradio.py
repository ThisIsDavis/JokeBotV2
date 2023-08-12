import gradio as gr
import os
from llama_index import SimpleDirectoryReader
# from llama_index.readers.file.docs_parser import PDFParser
from llama_index.readers.file.docs_reader import PDFReader
from llama_index.readers.schema.base import Document
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import ConversationChain, OpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from transformers import pipeline
from textblob import TextBlob

class GPTProcessing(object):
    
    def __init__(self, ui_obj):
        self.name = "Custom Data Processing with ChatGPT"
        self.description = "Add Custom PDF/txt Data Processing with ChatGPT"
        self.api_key = None
        self.index_folder = "indexData" #remember to put output index folder name
        self.compile_folder = "folder"
        self.ui_obj = ui_obj
        self.api_key_status = "Error: OpenAI API is not set"
        self.selected_index = None
        self.index_status = "Error: Index is not selected"
        self.index_setup_result = ".... no action..."
        self.OPENAI_API_KEY = "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"
        # Pre-trained voice recognition model.
        self.voice_recognition_model = pipeline("automatic-speech-recognition")

    def create_ui(self):
        with self.ui_obj:
            gr.Markdown('# MCS01 Joke Bot Application')
            with gr.Tab("Enter Folder Name"):
                text_input = gr.Textbox()
                text_output = gr.Textbox()
                text_button = gr.Button("Create Index!!!")
                text_button.click(self.build_the_bot, text_input, text_output)
            with gr.Tab("JokeBot"):
                chatbot = gr.Chatbot()
                message = gr.Textbox ("What is this document about?")
                message.submit(self.chat, [chatbot, message], chatbot)
                # Voice Recognition
                with gr.Row():
                    # Record audio
                    voice_recog = gr.Audio(source = "microphone", type="filepath")
                    # Button to start voice recognition
                    voice_recog_action = gr.Button("Keyword Voice Recognition")
            with gr.TabItem("User Joke Preference Learning"):
                with gr.Row():
                    gr.Textbox(label="Here are a list of jokes, please let us know which jokes speak to you the most! Put the joke numbers", placeholder= "1. List of jokes", interactive=False)
                with gr.Row():
                    gr.Textbox(label="Enter your joke preferences here:")
                    gr.Button("Submit")

    # Button to start recording voice and outputting it to the text box.
            voice_recog_action.click(
                self.transcribe_audio,
                [
                    voice_recog
                ], 
                [
                    message
                ]
            )
    
    # def create_ui(self):
    #     with self.ui_obj:
    #         gr.Markdown("Generating User Tailored Jokes From A Keyword")
    #         #now we set up the tabs in ui
    #         with gr.Tabs():
    #             #second tab item
    #             with gr.TabItem("Training/Fine-tuning with Custom Data: For testing purposes"):
    #                 with gr.Row():
    #                     source_data = gr.File(
    #                         label="PDF or Text files only",
    #                         file_count="single",
    #                         file_types=["file"])
    #                     index_setup_action = gr.Button("Create Index")
    #                 with gr.Row():
    #                     folder_name = gr.Textbox(label="Enter folder name:")
    #                     folder_submit = gr.Button("Submit")
    #                     folder_submit.click(self.set_folder_name, folder_name)
    #                     index_submit = gr.Button("Create Index for Folder")
    #                 with gr.Row():
    #                     index_setup_result_label = gr.Label(self.index_setup_result)
    #             #third tab : get user joke preference by allowing users to choose from a few joke categories.
    #             with gr.TabItem("User Joke Preference Learning"):
    #                 with gr.Row():
    #                     gr.Textbox(label="Here are a list of jokes, please let us know which jokes speak to you the most! Put the joke numbers", placeholder= "1. List of jokes", interactive=False)
    #                 with gr.Row():
    #                     gr.Textbox(label="Enter your joke preferences here:")
    #                     gr.Button("Submit")
    #             #third tab item
    #             with gr.TabItem("Query Custom Data: For testing purposes"):
    #                 with gr.Row():
    #                     #this part accepts queries
    #                     query_question = gr.Textbox(label="Enter your keyword", lines=5)
    #                     #button to get answer
    #                     query_data_action = gr.Button("Get Joke")
    #                 # Voice Recognition
    #                 with gr.Row():
    #                     # Record audio
    #                     voice_recog = gr.Audio(source = "microphone", type="filepath")
    #                     # Button to start voice recognition
    #                     voice_recog_action = gr.Button("Keyword Voice Recognition")   
    #                 with gr.Row():
    #                     #to show response
    #                     query_result_text = gr.Textbox(label="Query Result:", lines=10)
    #             #fourth tab item for generating malaysian jokes
    #             with gr.TabItem("Malaysian Jokes 4 U"):
    #                 with gr.Row():
    #                     #this part accepts queries
    #                     gr.Textbox(label="Enter your keyword prompt:", lines=5)
    #                     #button to get answer
    #                     gr.Button("Get Joke Lai")
    #                 with gr.Row():
    #                     #to show response
    #                     gr.Textbox(label="Joke Generated Liao", lines=10)
               
               
           
            # #setup second click button
            # index_submit.click(
            #     self.create_index,
            #     [
                
            #     ],
            #     [
            #         index_setup_result_label  #output
            #     ]
            # )

            # #setup second click button
            # index_setup_action.click(
            #     self.index_setup_process,
            #     [
            #         source_data    #input
            #     ],
            #     [
            #         index_setup_result_label  #output
            #     ]
            # )


            # query_data_action.click(
            #     self.get_answer_from_index,
            #     [
            #         query_question
            #     ],
            #     [
            #         query_result_text
            #     ]
            # )
            
            # # Button to start recording voice and outputting it to the text box.
            # voice_recog_action.click(
            #     self.transcribe_audio,
            #     [
            #         voice_recog
            #     ], 
            #     [
            #         query_question
            #     ]
            # )
            

    ############################## Helper Functions#################################################################
    def launch_ui(self):
        self.ui_obj.launch(share=True)

    #temporart
    def build_the_bot(self, input_text):
        max_input = 4096
        tokens = 200
        chunk_size = 600 #for LLM, we need to define chunk size
        max_chunk_overlap = 20
        promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)


        path = self.compile_folder
        text_list = [input_text]
        documents = [Document(t) for t in text_list]
        global index  #need to change this to file path later
        llmPredictor = LLMPredictor(llm=OpenAI(temperature=0.5, openai_api_key=self.OPENAI_API_KEY , model_name="gpt-3.5-turbo",max_tokens=tokens))

        service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=promptHelper)
        index = None
        return('Index saved successfull!!!')
     


    def chat(chat_history, user_input):
    
        bot_response = index.query(user_input)
        #print(bot_response)
        response = ""
        for letter in ''.join(bot_response.response): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
            response += letter + ""
            yield chat_history + [(user_input, response)]

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
                parser = PDFReader()
                extracted_pdf = parser.load_data(file)
                data_list.append(extracted_pdf)
                source_data = [Document(d) for d in data_list]
            elif file_extension[1] == '.txt':
                text_file_path = os.path.split(file)
                source_data = SimpleDirectoryReader(text_file_path[0]).load_data()

        return source_data
    
    #to index the file
    def index_setup_process(self, file):
        source_documents = self.load_source_data(file.name)
        status_message = "Error: Unable to create to source document index"
        if len(source_documents) > 0:
            source_index = GPTVectorStoreIndex.from_documents(source_documents)  #get the index from gpt api
            saved_file = self.save_index_document(source_index, file.name) #save the source index
            if saved_file is not None:
                status_message = "Success: The index is ready as [" + saved_file + "]" #this will be shown in the label row
        return status_message
    

    #to save the loaded data
    def save_index_document(self, source_index, out_file_name):
        try:
            final_out_file = os.path.basename(out_file_name.lower()) + ".json"
            final_out_file_path = os.path.join(os.getcwd(), self.index_folder, final_out_file)
            source_index.save_to_disk(final_out_file_path)
        except:
            final_out_file_path = None

        return final_out_file_path  #will see index.json in directory.
    

    #list the indexes created.
    def index_listing(self):
        index_path = os.path.join(os.getcwd(), self.index_folder)
        all_files = os.listdir(index_path)
        if len(all_files) == 0:
            return "No files"
        else:
            return all_files
        
    def setup_index_from_collection(self, index_name):
        if index_name is not None and len(index_name) > 0 and self.OPENAI_API_KEY is not None:
            index_path = os.path.join(os.getcwd(), self.index_folder, index_name)
            if os.path.exists(index_path):
                self.selected_index = GPTVectorStoreIndex.load_from_disk(index_path)
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
    
    
    def set_folder_name(self, folder_name):
        if folder_name and os.path.isdir(folder_name):
            self.compile_folder = folder_name
            print(self.compile_folder)
        print("Test2")

    
    def create_index(self):
        path = self.compile_folder
        max_input = 4096
        tokens = 200
        chunk_size = 600 #for LLM, we need to define chunk size
        max_chunk_overlap = 20
        
        #define prompt
        promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)
        
        #define LLM — there could be many models we can use, but in this example, let’s go with OpenAI model
        llmPredictor = LLMPredictor(llm=OpenAI(temperature=0.5, openai_api_key=self.OPENAI_API_KEY , model_name="gpt-3.5-turbo",max_tokens=tokens))

        #initialise conversation chain (memory)
        conversation = ConversationChain(llm=llmPredictor, memory= ConversationBufferWindowMemory(k=10))
        
        #load data — it will take all the .txtx files, if there are more than 1
        docs = SimpleDirectoryReader(path).load_data()

        service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor,prompt_helper=promptHelper)
        

        #create vector index
       
        vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs,service_context=service_context)
        final_out_file = "vectorIndex.json"
        final_out_file_path = os.path.join(os.getcwd(), self.index_folder, final_out_file)
        vectorIndex.save_to_disk(final_out_file_path)

        #generates status message
        if final_out_file_path is not None:
                status_message = "Success: The index is ready as [" + final_out_file_path + "]" #this will be shown in the label row
        return status_message

    # Voice recognition.
    def transcribe_audio(self, audio):
        # Recognise audio input.
        text = self.voice_recognition_model(audio)["text"]
        
        # If no audio received, give error.
        if len(text) == 0 or text.isspace():
            result = "Error: Unable to recognise audio"
        # Else split the text and returned only the first word.
        else:
            text = str(text.split()[0])
            # Autocorrect any spelling mistakes.
            result = TextBlob(text)
            result = result.correct()
            
        return result
    

if __name__ == '__main__':
    my_app = gr.Blocks()
    #this will call our program with gr.Block from gradio
    gradio_ui = GPTProcessing(my_app)
    # Create UI
    gradio_ui.create_ui()
    #to launch the ui
    gradio_ui.launch_ui()


