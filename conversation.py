from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

OPENAI_API_KEY = "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"
# Initialize the language model
llm = ChatOpenAI(temperature=0.5, openai_api_key=OPENAI_API_KEY, model="gpt-3.5-turbo")

# Define the prompt template
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'"""
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
)

# Initialize the conversation chain with buffer window memory
conversation = ConversationChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=3, return_messages=True),
    chain_type_kwargs={"return_messages": True},
)

# Predict a response for a given input
response = conversation.predict(input="Tell me a joke")
