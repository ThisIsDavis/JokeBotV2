import gradio as gr
import os, os.path
import openai
import speech_recognition as sr
import random
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, load_index_from_storage, StorageContext, SimpleDirectoryReader
from langchain import ConversationChain, OpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from utils import *


class GPTProcessing(object):
    
    def __init__(self, ui_obj):
        self.ui_obj = ui_obj
        self.tag_memory = []
        self.upvote_prompts = []
        self.downvote_prompts = []
        self.input = None
        self.output = None
        self.OPENAI_API_KEY = ""
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        openai.api_key = self.OPENAI_API_KEY

        # Get the number of categories of jokes in the jokes folder.
        self.scraped_jokes = [name for name in os.listdir(os.path.join(os.getcwd(), "jokes"))]
        self.num_of_categories = len(self.scraped_jokes)
        
        # Saved joke preferences by the user.
        self.user_joke_preferences = []
        self.count = 1

    def create_ui(self):
        # Call the get_random_jokes method to get a list of randomly selected jokes.
        self.preference_jokes = self.get_random_jokes()
        
        with self.ui_obj:
            gr.Markdown('# MCS01 Joke Bot Application')
            with gr.Tab("JokeBot"):
                chatbot = gr.components.Chatbot(label='Finetuned Joke Machine', height = 600)
                message = gr.components.Textbox (label = 'User Keyword')
                state = gr.State()
                message.submit(self.message_and_history, inputs=[message, state],  outputs=[chatbot, state])
                # Voice Recognition
                with gr.Row():
                    # Record audio and output the audio filepath.
                    voice_recog = gr.Audio(source = "microphone", type = "filepath")
                    # Button to start voice recognition
                    voice_recog_action = gr.Button("Keyword Voice Recognition")
                # Buttons Galore
                with gr.Row():
                    upvote_btn = gr.Button(value="👍  Upvote")
                    downvote_btn = gr.Button(value="👎  Downvote")
                    regenerate_btn = gr.Button(value="🔄  Regenerate")
                    clear_btn = gr.Button(value="🗑️  Clear prompt")
                    
                    upvote_btn.click(lambda: self.tag_response(1, None), inputs=[], outputs=[])
                    downvote_btn.click(lambda: self.tag_response(0, None), inputs=[], outputs=[])
                    regenerate_btn.click(self.regenerate, inputs=[state],  outputs=[chatbot, state])
                    clear_btn.click(lambda: message.update(""), inputs=[], outputs=[message])
                # Who to recommend?
                with gr.Row():
                    def on_send_btn_click(input):
                        self.tag_response(None, input)

                    recommend_textbox = gr.components.Textbox(label='Who would you recommend the current joke to?')
                    send_btn = gr.Button(value='Enter')
                    send_btn.click(on_send_btn_click, inputs=[recommend_textbox], outputs=[])
            with gr.TabItem("User Joke Preference Learning"):
                with gr.Row():
                    # gr.Textbox(label="Here are a list of jokes, please let us know which jokes speak to you the most! Put the joke numbers", placeholder= "1. List of jokes", interactive=False)
                    joke_preferences = gr.CheckboxGroup(self.preference_jokes, label = "List of Jokes", info = "Please select which jokes speaks to you the most!")
                    joke_preferences_action = gr.Button("Submit")
                with gr.Row():
                    selected_joke_preferences = gr.Textbox(label = "Selected Joke Preferences:", info = "List of jokes that you find funny!",placeholder = "No selected joke preferences yet!", interactive = False)
                    selected_joke_preferences_action = gr.Button("Clear Preferences", scale = 0.5)

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
            
            # Button to save all selected joke preferences and display them back to the user.
            joke_preferences_action.click(
                self.save_joke_preference,
                [
                    joke_preferences, selected_joke_preferences
                ], 
                [
                    selected_joke_preferences
                ]
            )
            
            selected_joke_preferences_action.click(
                self.clear_joke_preference,
                [],
                [
                    selected_joke_preferences
                ]
            )

    ############################## Helper Functions#################################################################
    def launch_ui(self):
        self.ui_obj.queue().launch(share=True)

    def api_calling(self, prompt):
        big_prompt = "Give me a joke about " + prompt
        print(prompt)

        # If there is upvoted responses and downvoted responses
        if len(self.upvote_prompts) > 0 and len(self.downvote_prompts) > 0:
            up_pr = ", ".join(self.upvote_prompts)
            do_pr = ", ".join(self.downvote_prompts)
            big_prompt += " with similar jokes to " + up_pr + " and not similar to" + do_pr
        # If there is only 1 downvoted response
        elif len(self.downvote_prompts) > 0:
            do_pr = ", ".join(self.downvote_prompts)
            big_prompt += " that are not similar to " + do_pr
        # If there is only 1 upvoted response
        elif len(self.upvote_prompts) > 0:
            up_pr = ", ".join(self.upvote_prompts)
            big_prompt += " with similar jokes to " + up_pr

        completions = openai.Completion.create(
            engine="davinci:ft-monash-university-malaysia-2023-08-16-12-29-02",
            prompt=big_prompt,
            max_tokens=5,
            temperature=0.7,
        )
        print(big_prompt)
        message = completions.choices[0].text
        return message

    def message_and_history(self, input, history):
        self.input = input
        history = history or []
        # s = list(sum(history, ()))
        # s.append(input)
        # inp = ' '.join(s)
        self.output = self.api_calling(input)
        history.append((input, self.output))
        self.tag_memory.append([None, None])
        return history, history
    
    def regenerate(self, history):
        if self.input is not None:
            return self.message_and_history(self.input, history)
        else:
            pass

    def tag_response(self, vote, recommendation):
        # Ensure there is a response before properly voting and recommending
        if len(self.tag_memory) > 0:
            # Get the current response
            last_response = self.tag_memory[-1]
            # If there is a recommendation
            if recommendation is not None and len(recommendation) > 0:
                last_response[2] = recommendation
            # If there is a downvote
            elif vote == 0:
                last_response[0] = vote
                self.downvote_prompts.append(self.output)
            # If there is an upvote
            elif vote == 1:
                last_response[0] = vote
                self.upvote_prompts.append(self.output)
            print(self.tag_memory)
        else:
            pass

    # temporart (Build the bot no longer need #Davis)
    # def build_the_bot(self, input_text):
    #     if os.path.exists(input_text):
    #         print("Passed")

    #     max_input = 4096
    #     tokens = 200
    #     chunk_size = 600 #for LLM, we need to define chunk size
    #     max_chunk_overlap = 1
    #     promptHelper = PromptHelper(max_input,tokens,max_chunk_overlap,chunk_size_limit=chunk_size)
    #     path = self.compile_folder
    #     docs = SimpleDirectoryReader(input_dir=path).load_data()
    #     llmPredictor = LLMPredictor(llm=OpenAI(temperature=0.5, openai_api_key=self.OPENAI_API_KEY , model_name="gpt-3.5-turbo", max_tokens=tokens))

    #     service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=promptHelper)

    #     vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
    #     final_out_file = "vectorIndex.json"
    #     self.saved_path = os.path.join(os.getcwd(), self.index_folder, final_out_file)
    #     vectorIndex.storage_context.persist(persist_dir=self.saved_path)
        
    #     return('Index saved successfull!!!')

    # def chat(self, chat_history, user_input):
    #     # self.saved_path = os.path.join(os.getcwd(), self.index_folder, "vectorIndex.json")
    #     # storage_context = StorageContext.from_defaults(persist_dir=self.saved_path)
    #     # self.selected_index = load_index_from_storage(storage_context)

    #     # chat_engine = self.selected_index.as_chat_engine(verbose=True)
    #     # bot_response = chat_engine.stream_chat(user_input)

    #     #we no longer use Vector Index, now we use our openAI fine tuned model 
    #     # openai.api_key = os.getenv("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")
    #     bot_response = openai.Completion.create(
    #         model="davinci:ft-monash-university-malaysia-2023-08-16-12-29-02",
    #         prompt=user_input,
    #         max_tokens=50,
    #         temperature=0.7,
    #         stream=True)

    #     response = ""
    #     for letter in ''.join(bot_response): #[bot_response[i:i+1] for i in range(0, len(bot_response), 1)]:
    #         response += letter + ""
    #         yield chat_history + [(user_input, response)]

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
    
    def get_random_jokes(self):
        """
        A method which randomly selects a joke category from the webscraped jokes within the jokes folder, then randomly select one
        joke inside said category. Will then append it all into a list and return it.
        :Input:
            None
        :Output:
            jokes: A list of jokes randomly taken from the webscraped jokes.
        """
        categories = []     # Empty list which will hold the indexes of the joke text files.
        jokes = []          # Empty list which will contain the random jokes.
        
        # Randomly select a joke category and append it to the categories list 5 times.
        i = 0
        while i < 5:
            # Generate a random number from 0 to the maximum number of categories available
            random_number = random.randint(0, self.num_of_categories)
            # If the random number has not been generated before/not in the categories list, append it in and increment i by one.
            if random_number not in categories:
                categories.append(random_number)
                i += 1
        
        # Loop through the category list and grab a random joke from that joke category text file.
        for category in categories:
            # Open up the random joke text file and select a random joke.
            with open(os.path.join("jokes/", self.scraped_jokes[category]), 'r', encoding = "utf-8") as f:
                file = f.readlines()        # Read the joke file contents.  
                
                # Get the joke string based on a randomly selected number from 3 to the number of jokes + 1. 
                # As each joke is separated by a new line, only index odd numbers with step 1.
                random_number = random.randrange(3, int(file[-1]) + 1, 2) - 1
                random_joke = file[random_number]
            
                jokes.append(random_joke)        # Append the selected joke into the jokes list.
                
        return jokes
    
    def save_joke_preference(self, jokes, selected_jokes) -> str:
        """
        Takes in a list of strings of jokes which the user has selected, saving it.
        :Input:
            jokes: The list of jokes the user has selected/preferred.
            selected_jokes: A string of selected jokes based on the textbox displayed. Initially empty.
        :Output:
            joke_str: A string containing all the jokes the user selected/preferred.
        """
        # Append the list of selected jokes to self.user_joke_preferences to save it.
        self.user_joke_preferences.append(jokes)
        # Set the joke_str variable to the string of selected joke preferences.
        joke_str = selected_jokes
        
        # Loop through all the selected joke preferences and concatenate it to the joke_str to be displayed back to the user.
        for joke in jokes:
            # If the textbox is not empty, add a newline before adding the joke.
            if len(joke_str) != 0:
                joke_str += "\n{0}. {1}".format(self.count, joke)
            # Else if the textbox is initally empty, do not add a new line on the top.
            else:
                joke_str += "{0}. {1}".format(self.count, joke)
            
            # Increment count by one.
            self.count += 1
        
        return joke_str     # Return the string of selected jokes.
    
    def clear_joke_preference(self) -> str:
        """
        Clear saved user joke preferences by resetting the instance variables and clearing the TextBox.
        :Input:
            None
        :Output:
            selected_jokes: An empty string to represent the TextBox being cleared.
        """
        self.user_joke_preferences = []     # Clear the saved user joke preferences.
        self.count = 1                      # Reset the count to one.
        
        return ""   # Return an empty string.


if __name__ == '__main__':
    my_app = gr.Blocks()
    #this will call our program with gr.Block from gradio
    gradio_ui = GPTProcessing(my_app)
    # Create UI
    gradio_ui.create_ui()
    #to launch the ui
    gradio_ui.launch_ui()


