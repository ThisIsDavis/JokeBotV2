import gradio as gr
import os
import speech_recognition as sr
from llama_index import SimpleDirectoryReader
# from llama_index.readers.file.docs_parser import PDFParser
from llama_index.readers.file.docs_reader import PDFReader
from llama_index.readers.schema.base import Document
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain import ConversationChain, OpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


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
                    # Record audio and output the audio filepath.
                    voice_recog = gr.Audio(source = "microphone", type = "filepath")
                    # Button to start voice recognition
                    voice_recog_action = gr.Button("Keyword Voice Recognition")
            with gr.TabItem("User Joke Preference Learning"):
                with gr.Row():
                    gr.Textbox(label="Here are a list of jokes, please let us know which jokes speak to you the most! Put the joke numbers", placeholder= "1. List of jokes", interactive=False)
                with gr.Row():
                    gr.Textbox(label="Enter your joke preferences here:")
                    gr.Button("Submit")

    # Button to start recording voice and outputting it to the message text box.
            voice_recog_action.click(
                self.transcribe_audio,
                [
                    voice_recog
                ], 
                [
                    message
                ]
            )

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

    # Voice recognition.
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Takes in the filepath where the audio is stored and use an Automatic Speech Recognition Model to transcribe the audio into text.
        Will take the first word of the whole transcribed sentence.
        :Input:
            audio_path: The filepath where the audio is stored.
        :Output:
            text: The transcribed text based on the input audio.
        """
        # Initalise a Recognizer instance.
        recogniser = sr.Recognizer()
        
        # Open the audio file based on inputted filepath an initalise it to source variable.
        with sr.AudioFile(audio_path) as source:
            # Extract audio data from the source file.
            audio = recogniser.record(source)
            
            # If there's audio, recognise the speech using Google Speech Recognition and return the first keyword transcribed.
            try:
                text = recogniser.recognize_google(audio)   # Transcribe the audio into text.
                text = str(text.split()[0]).lower()         # Extract the first word of the text and then turn it to lowercase.
                return text
            
            # Speech is unintelligible or other errors, return error text.
            except:
                text = "Error: Unable to recognise audio"
                return text
    

if __name__ == '__main__':
    my_app = gr.Blocks()
    #this will call our program with gr.Block from gradio
    gradio_ui = GPTProcessing(my_app)
    # Create UI
    gradio_ui.create_ui()
    #to launch the ui
    gradio_ui.launch_ui()


