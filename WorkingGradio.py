import gradio as gr
import os, os.path
import openai
import speech_recognition as sr
import random
import re

class GPTProcessing(object):
    
    def __init__(self, ui_obj):
        # For normal jokes
        self.ui_obj = ui_obj
        self.profanity_list = ["fuck", "shit", "bitch", "ass", "sex", "motherfucker", "asshole", "dick", "cunt", "tits", "piss", "twat", "penis", "vagina", "pussy", "boobs"]  # List of profane words to be filtered
        self.tag_memory = []
        self.upvote_prompts = []
        self.downvote_prompts = []
        self.input = None
        self.output = None

        # For Malaysian jokes
        self.tag_memory_my = []
        self.upvote_prompts_my = []
        self.downvote_prompts_my = []
        self.input_my = None
        self.output_my = None

        self.OPENAI_API_KEY = ""
        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        openai.api_key = self.OPENAI_API_KEY

        # Get the number of categories of jokes in the jokes folder.
        self.scraped_jokes = [name for name in os.listdir(os.path.join(os.getcwd(), "jokes"))]
        self.num_of_categories = len(self.scraped_jokes)
        self.limit = 1      # The limits of preferred jokes.
        
        # Saved joke preferences by the user.
        self.user_joke_preferences = []
        self.user_joke_categories = []
        self.count = 1

    def create_ui(self):
        # Call the get_random_jokes method to get a list of randomly selected jokes.
        self.preferred_joke_categories = self.get_random_jokes()
        
        with self.ui_obj:
            gr.Markdown('# MCS01 Joke Bot Application')
            with gr.TabItem("User Joke Preference Learning"):
                with gr.Row():
                    # gr.Textbox(label="Here are a list of jokes, please let us know which jokes speak to you the most! Put the joke numbers", placeholder= "1. List of jokes", interactive=False)
                    joke_preferences = gr.CheckboxGroup(self.preferred_joke_categories, label = "List of Jokes", info = "Please select at most ONE joke which speaks to you the most!")
                with gr.Row():
                    joke_preferences_action = gr.Button("Submit")
                    joke_refresh_action = gr.Button("Refresh Jokes")
                with gr.Row():
                    selected_joke_preferences = gr.Textbox(label = "Selected Joke Preferences:", info = "List of jokes that you find funny!", placeholder = "No selected joke preferences yet!", interactive = False, scale = 3)
                    selected_joke_preferences_action = gr.ClearButton([joke_preferences], value = "Clear Preferences", scale = 1)

            with gr.Tab("JokeBot"):
                chatbot = gr.components.Chatbot(label='Finetuned Joke Machine', height = 600)  
                message = gr.components.Textbox (label = 'User Keyword')
                state = gr.State()

                # Sends the user input keyword to generate a joke
                message.submit(self.message_and_history, inputs=[message, state],  outputs=[chatbot, state])

                # Voice Recognition
                with gr.Row():
                    # Record audio and output the audio filepath.
                    voice_recog = gr.Microphone(type = "filepath")
                    voice_recog.stop_recording(self.transcribe_audio, inputs = [voice_recog], outputs = [message])
                
                # Buttons Galore
                with gr.Row():
                    upvote_btn = gr.Button(value="👍  Upvote")  # Upvote Button
                    downvote_btn = gr.Button(value="👎  Downvote")  # Downvote Button
                    clear_btn = gr.Button(value="🗑️  Clear Keyword")  # Clear Keyword Button
                    refresh_btn = gr.Button(value="🔄  Refresh")  # Refresh Button
                    
                    # Logic when one of the buttons is clicked
                    upvote_btn.click(lambda: self.tag_response(1, None), inputs=[], outputs=[])  
                    downvote_btn.click(lambda: self.tag_response(0, None), inputs=[], outputs=[])
                    clear_btn.click(lambda: message.update(""), inputs=[], outputs=[message])
                    refresh_btn.click(self.refresh, inputs=[],  outputs=[])

                # Who to recommend?
                with gr.Row():
                    def on_send_btn_click(input):
                        self.tag_response(None, input)

                    recommend_textbox = gr.Dropdown(["Parents", "Siblings", "Relatives", "Friends", "Colleagues"], allow_custom_value = True, label = "Who would you recommend the current joke to?" , interactive = True, scale = 4)
                    send_btn = gr.ClearButton([recommend_textbox], value = "Submit Feedback", scale = 1)
                    send_btn.click(on_send_btn_click, inputs=[recommend_textbox], outputs=[])

            # Malaysian Jokes Section
            with gr.Tab("Malaysian JokeBot 🇲🇾"):
                chatbot_my = gr.components.Chatbot(label='Finetuned Malaysian Joke Machine', height = 600)
                message_my = gr.components.Textbox (label = 'User Keyword')
                state_my = gr.State()
                message_my.submit(self.message_and_history_my, inputs=[message_my, state_my],  outputs=[chatbot_my, state_my])
                
                # Voice Recognition
                with gr.Row():
                    # Record audio and output the audio filepath.
                    voice_recog_my = gr.Microphone(type = "filepath")
                    voice_recog_my.stop_recording(self.transcribe_audio, inputs = [voice_recog_my], outputs = [message_my])
                    
                # Buttons Galore
                with gr.Row():
                    upvote_btn_my = gr.Button(value="👍  SHIOK")  # Upvote Button
                    downvote_btn_my = gr.Button(value="👎  Potong Stim")  # Downvote Button
                    clear_btn_my = gr.Button(value="🗑️  Clear Keyword")  # Clear Keyword Button
                    refresh_btn_my = gr.Button(value="🔄  Refresh")  # Refresh Button
                    
                    upvote_btn_my.click(lambda: self.tag_response_my(1, None), inputs=[], outputs=[])
                    downvote_btn_my.click(lambda: self.tag_response_my(0, None), inputs=[], outputs=[])
                    clear_btn_my.click(lambda: message_my.update(""), inputs=[], outputs=[message_my])
                    refresh_btn_my.click(self.refresh, inputs=[],  outputs=[])

                # Who to recommend?
                with gr.Row():
                    def on_send_btn_click_my(input):
                        self.tag_response_my(None, input)

                    recommend_textbox_my = gr.Dropdown(["Ibu bapa", "Siblings", "Relatives", "Kawan-kawan", "Colleagues"], allow_custom_value = True, label = "Who you nak recommend joke ini?" , interactive = True, scale = 4)
                    send_btn_my = gr.ClearButton([recommend_textbox_my], value = "Submit Feedback", scale = 1)
                    send_btn_my.click(on_send_btn_click_my, inputs=[recommend_textbox_my], outputs=[])

            # Button to save all selected joke preferences and display them back to the user.
            joke_preferences_action.click(
                self.save_joke_preference,
                [
                    joke_preferences
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
            
            joke_refresh_action.click(
                self.refresh_joke_preference,
                [],
                [
                    joke_preferences
                ]
            )

    ############################## Helper Functions#################################################################
    def launch_ui(self) -> None:        
        """
        A method that launches the Gradio UI website
        """
        self.ui_obj.queue().launch(share=True)

    def profanity_filter(self, text: str) -> str:
        """
        Filter out any profanities in the joke
        :Input:
            text: The joke given
        :Output:
            The profanity filtered joke
        """
        
        def replace(word: str) -> str:
            """
            Keep the first letter of the swear word, replace the rest with asterisks
            :Input:
                word: The swear word
            :Output:
                A censored swear word
            """
            return word.group(0)[0] + "*" * (len(word.group(0)) - 1)
    
        # Regular expression to find profanity in the text
        regex = r"\b(" + "|".join(map(re.escape, self.profanity_list)) + r")\b"
        return re.sub(regex, replace, text, flags=re.IGNORECASE)
    
    def create_message(self, complete_prompt: str) -> str:
        """
        Uses the created prompt and generates a joke 
        :Input:
            complete_prompt: The prompt to be used for the GPT-3.5-Turbo model 
        :Output:
            message: The joke generated
        """
        completions = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:monash-university-malaysia::7rREegcc",
            messages=[
                {"role": "system", "content": "JokeBot is a chatbot that tells funny jokes from given keywords"},
                {"role": "user", "content": complete_prompt}
            ],
            max_tokens=250,
            temperature=0.65,
        )

        message = completions.choices[0].message.content  # Get the joke
        return message

    def api_calling(self, prompt: str) -> str:
        """
        Creates the prompt based on the user input keyword, previously upvoted and downvoted jokes. Then generates a joke using the
        prompt created. 
        :Input:
            prompt: The user input keyword
        :Output:
            message: The joke generated from the user input keyword
        """
        if len(self.user_joke_preferences) == 0:
            # Prompt engineering the prompt
            big_prompt = f"Give me another type of joke about the word: {prompt}."
        else:
            big_prompt = f"Give me another joke about the word: {prompt} and related to: {self.user_joke_categories[0]}"
            
        # Ensure there is a response before crafting voted prompts
        if len(self.tag_memory) > 0:
            # Get the previous response
            last_response = self.tag_memory[-1]
            
            # If previous response is downvoted 
            if last_response[0] == 0:
                big_prompt = f"The joke was not funny. Please give me another type of joke about the word: {prompt}."
            # If previous response is upvoted 
            elif last_response[0] == 1:
                # Get the last upvoted joke.
                up_pr = self.upvote_prompts[-1]
                # Create the big prompt based on the upvoted joke.
                big_prompt = f"Learn from this joke: {up_pr} and give me another joke about the word: {prompt}."
                
        print(big_prompt)
        message = self.create_message(big_prompt)  # Get the joke
           
        message = self.profanity_filter(message)  # Filter out any profanities after the prompt is in the message
        return message

    def message_and_history(self, input: str, history):
        """
        Create a chat history if it doesn't exist and generates a joke based on the user input keyword. Then display the chat on the
        chatbot
        :Input:
            input: The user input keyword
            history: The state of the chat history as an array of chats 
        :Output:
            history, history: The state of the chat history after generating a joke
        """
        self.input = input.split(' ')[0]  # Get the first word of the input
        
        # Only generate joke if keyword includes a character
        if self.input != "":
            history = history or []  # Create chat history list if doesn't exist or use existing chat history
            self.create_feedback(history, self.tag_memory)  # Append the previous chat and its feedback to a file
            self.output = self.api_calling(self.input)  # Generate the joke
            history.append((self.input, self.output))  # Append the prompt and joke to the chatbot display
            self.tag_memory.append([None, None])  # Create a tag memory for the new joke
        return history, history
    
    def refresh(self) -> None:
        """
        Refresh the entire preference by deleting them
        """
        # For normal jokes
        self.tag_memory.clear()
        self.upvote_prompts.clear()
        self.downvote_prompts.clear()
        self.input = None
        self.output = None

        # For Malaysian jokes
        self.tag_memory_my.clear()
        self.upvote_prompts_my.clear()
        self.downvote_prompts_my.clear()
        self.input_my = None
        self.output_my = None
    

    def tag_response(self, vote: int, recommendation: str) -> None:
        """
        Tags the current existing prompt and joke with a vote and recommendation
        :Input:
            vote: An integer representing the vote (0 -> downvote, 1 -> upvote)
            recommendation: A string of who the user would recommend the joke to 
        """
        # Ensure there is a response before properly voting and recommending
        if len(self.tag_memory) > 0:
            # Get the current response
            last_response = self.tag_memory[-1]
            # If there is a recommendation, set the recommendation
            if recommendation is not None and len(recommendation) > 0:
                last_response[1] = recommendation
            # If there is a downvote, set the vote to 0 and append the output to downvote prompts
            elif vote == 0:
                last_response[0] = vote
                self.downvote_prompts.append(self.output)
            # If there is an upvote, set the vote to 1 and append the output to upvote prompts
            elif vote == 1:
                last_response[0] = vote
                self.upvote_prompts.append(self.output)
        else:
            pass

    def create_feedback(self, chat_lst, tag_list) -> None:
        """
        Appends the previous chat and tagged information into a file for data collection
        :Input:
            chat_lst: The chat history as an array
            tag_list: The tagged information of the chat history as an array 
        """
        # Checks if there is an existing joke
        if len(chat_lst) and len(tag_list):

            # If yes, get the last joke's details
            last_message = chat_lst[-1]
            last_feedback = tag_list[-1]

            # Write and append to feedback.txt file
            with open('feedback.txt', 'a') as f:
                f.write(f'Prompt: {last_message[0]}\n')  # Write prompt
                f.write(f'Joke: {last_message[1]}\n')   # Write joke

                # Checks what is the vote
                if last_feedback[0] == 0:
                    f.write('Vote: Downvote\n')  # Write vote as downvote
                elif last_feedback[0] == 1:
                    f.write('Vote: Upvote\n')  # Write vote as upvote
                else:
                    f.write('Vote: None\n')  # Write vote as None

                f.write(f'Recommend: {last_feedback[1]}\n\n')  # Write recommendation
        else:
            pass

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
        # Initialise a Recognizer instance.
        recogniser = sr.Recognizer()
        
        # Open the audio file based on inputted filepath an initialise it to source variable.
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
            random_number = random.randint(0, self.num_of_categories - 1)
            # If the random number has not been generated before/not in the categories list, append it in and increment i by one.
            if random_number not in categories:
                categories.append(random_number)
                i += 1
        
        # Loop through the category list and grab a random joke from that joke category text file.
        for category in categories:
            # Get the name of the category.
            category_name = self.scraped_jokes[category]
            # Open up the random joke text file and select a random joke.
            with open(os.path.join("jokes/", category_name), 'r', encoding = "utf-8") as f:
                file = f.readlines()        # Read the joke file contents.  
                
                # Get the joke string based on a randomly selected number from 3 to the number of jokes + 1. 
                # As each joke is separated by a new line, only index odd numbers with step 1.
                random_number = random.randrange(3, int(file[-1]) + 1, 2) - 1
                random_joke = file[random_number]
            
                jokes.append((random_joke, category_name[:-4]))      # Append the selected joke into the jokes list.
                
        return jokes
    
    def save_joke_preference(self, jokes) -> str:
        """
        Takes in a list of strings of jokes which the user has selected, saving it.
        :Input:
            jokes: The list of jokes the user has selected/preferred.
            selected_jokes: A string of selected jokes based on the textbox displayed. Initially empty.
        :Output:
            joke_str: A string containing all the jokes the user selected/preferred.
        """
        # Loop through each selected joke to add it the to user preferences list. Will not allow duplicated jokes from being added.
        for joke in jokes:
            # Prevent addition of duplicated jokes and only allow the addition of jokes up to a set limit.
            if (joke[1] not in self.user_joke_categories) and (len(self.user_joke_categories) < self.limit):
                # Append the list of selected jokes to self.user_joke_preferences to save it.
                self.user_joke_preferences.append(joke[0])
                self.user_joke_categories.append(joke[1])
                        
        joke_str = ""       # Create empty string to store all user-preffered jokes.
        self.count = 1      # Set count to 0 at start.
        
        # Loop through all the selected joke preferences and concatenate it to the joke_str to be displayed back to the user.
        for joke in self.user_joke_preferences:
            # If the textbox is not empty, add a newline before adding the joke.
            if len(joke_str) != 0:
                joke_str += "\n{0}. {1}".format(self.count, joke)
            # Else if the textbox is initially empty, do not add a new line on the top.
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
        # Remove all common jokes in the upvote prompt and user joke preferences list.
        self.upvote_prompts = list(set(self.user_joke_preferences)^set(self.upvote_prompts))
        self.upvote_prompts_my = list(set(self.user_joke_preferences)^set(self.upvote_prompts_my))
        self.user_joke_preferences = []     # Clear the saved user joke preferences.
        self.user_joke_categories = []      # Clear the saved user joke categories.
        self.count = 1                      # Reset the count to one.

        return ""   # Return an empty string.
    
    def refresh_joke_preference(self):
        """
        Refreshes all jokes for user preference with another random 5 jokes.
        :Input:
            None
        :Output:
            updated_joke_preferences: The updated CheckboxGroup element of all jokes displayed.
        """
        self.preferred_joke_categories = self.get_random_jokes()
        
        updated_joke_preferences = gr.CheckboxGroup.update(choices = self.preferred_joke_categories)

        return updated_joke_preferences
    
    ############################## Malaysian Jokes Helper Functions #################################################################

    def create_message_my(self, complete_prompt_my: str) -> str:
        """
        Uses the created prompt and generates a joke 
        :Input:
            complete_prompt_my: The prompt to be used for the GPT-3.5-Turbo model 
        :Output:
            message_my: The joke generated
        """
        completions = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:monash-university-malaysia::7yCzUcJq",
            messages=[
                {"role": "system", "content": "JokeBot is a chatbot that tells funny jokes in Malaysian context from given keywords"},
                {"role": "user", "content": complete_prompt_my}
            ],
            max_tokens=1000,
            temperature=0.65,
        )

        message_my = completions.choices[0].message.content  # Get the joke
        return message_my
    
    def api_calling_my(self, prompt_my: str) -> str:
        """
        FOR MALAYSIAN JOKES
        Creates the prompt based on the user input keyword, previously upvoted and downvoted jokes. Then generates a joke using the
        prompt created. 
        :Input:
            prompt_my: The user input keyword
        :Output:
            message_my: The joke generated from the user input keyword
        """
        if len(self.user_joke_preferences) == 0:
            # Prompt engineering the prompt
            big_prompt_my = f"Give me another type of joke about the word: '{prompt_my}' in Malaysian context."
        else:
            big_prompt_my = f"Give me another type of joke about the word: '{prompt_my}' in Malaysian context and related to: {self.user_joke_categories[0]}"
        
        # Ensure there is a response before crafting voted prompts
        if len(self.tag_memory_my) > 0:
            last_response_my = self.tag_memory_my[-1]
            
            # If previous response is downvoted
            if last_response_my[0] == 0:
                big_prompt_my = f"The joke was not funny. Please give me another type of joke about the word: '{prompt_my}' in Malaysian context."
            # If previous response is upvoted
            elif last_response_my[0] == 1:
                up_pr_my = self.upvote_prompts_my[-1]
                big_prompt_my = f"Learn from this joke: '{up_pr_my}' and give me another joke about the word: '{prompt_my}' in Malaysian context."
            
        print(big_prompt_my)
        message_my = self.create_message_my(big_prompt_my)  # Get the joke
                    
        message_my = self.profanity_filter(message_my)  # Filter out any profanities after the prompt is in the message
        return message_my
    
    def message_and_history_my(self, input: str, history_my):
        """
        FOR MALAYSIAN JOKES
        Create a chat history if it doesn't exist and generates a joke based on the user input keyword. Then display the chat on the
        chatbot
        :Input:
            input: The user input keyword
            history_my: The state of the chat history as an array of chats 
        :Output:
            history_my, history_my: The state of the chat history after generating a joke
        """
        self.input_my = input.split(' ')[0]  # Get the first word of the input
        
        # Only generate joke if keyword includes a character
        if self.input_my != "":
            history_my = history_my or []  # Create chat history list if doesn't exist or use existing chat history
            self.create_feedback_my(history_my, self.tag_memory_my)  # Append the previous chat and its feedback to a file
            self.output_my = self.api_calling_my(self.input_my)  # Generate the joke
            history_my.append((self.input_my, self.output_my))  # Append the prompt and joke to the chatbot display
            self.tag_memory_my.append([None, None])  # Create a tag memory for the new joke
        return history_my, history_my

    def tag_response_my(self, vote_my: int, recommendation_my: str) -> None:
        """
        FOR MALAYSIAN JOKES
        Tags the current existing prompt and joke with a vote and recommendation
        :Input:
            vote_my: An integer representing the vote (0 -> downvote, 1 -> upvote)
            recommendation_my: A string of who the user would recommend the joke to 
        """
        # Ensure there is a response before properly voting and recommending
        if len(self.tag_memory_my) > 0:
            # Get the current response
            last_response_my = self.tag_memory_my[-1]
            # If there is a recommendation, set the recommendation
            if recommendation_my is not None and len(recommendation_my) > 0:
                last_response_my[1] = recommendation_my
            # If there is a downvote, set the vote to 0 and append the output to downvote prompts
            elif vote_my == 0:
                last_response_my[0] = vote_my
                self.downvote_prompts_my.append(self.output_my)
            # If there is an upvote, set the vote to 1 and append the output to upvote prompts
            elif vote_my == 1:
                last_response_my[0] = vote_my
                self.upvote_prompts_my.append(self.output_my)
        else:
            pass
    
    def create_feedback_my(self, chat_lst_my, tag_list_my) -> None:
        """
        FOR MALAYSIAN JOKES
        Appends the previous chat and tagged information into a file for data collection
        :Input:
            chat_lst_my: The chat history as an array
            tag_list_my: The tagged information of the chat history as an array 
        """
        # Checks if there is an existing joke
        if len(chat_lst_my) and len(tag_list_my):

            # If yes, get the last joke's details
            last_message_my = chat_lst_my[-1]
            last_feedback_my = tag_list_my[-1]

            # Write and append to feedback.txt file
            with open('feedback_my.txt', 'a') as f:
                f.write(f'Prompt: {last_message_my[0]}\n')  # Write prompt
                f.write(f'Joke: {last_message_my[1]}\n')   # Write joke

                # Checks what is the vote
                if last_feedback_my[0] == 0:
                    f.write('Vote: Downvote\n')  # Write vote as downvote
                elif last_feedback_my[0] == 1:
                    f.write('Vote: Upvote\n')  # Write vote as upvote
                else:
                    f.write('Vote: None\n')  # Write vote as None

                f.write(f'Recommend: {last_feedback_my[1]}\n\n')  # Write recommendation
        else:
            pass
    

if __name__ == '__main__':
    my_app = gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="pink"))
    #this will call our program with gr.Block from gradio
    gradio_ui = GPTProcessing(my_app)
    # Create UI
    gradio_ui.create_ui()
    #to launch the ui
    gradio_ui.launch_ui()
    