# import os
# import openai
# # openai.api_key = os.getenv('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')


# def chat_completion(messages: list) -> list[str]:
#     # try: 
#     bot_response = openai.Completion.create(
#         model="davinci:ft-monash-university-malaysia-2023-08-16-12-29-02",
#         messages=messages,
#         max_tokens=10,
#         temperature=0.7,
#         stream=True
#     )
#     collected_messages = []
#     for chunk in bot_response:
#         delta = chunk['choices'][0]['text']
#         print(chunk)
#         if 'content' in delta.keys():
#             collected_messages.append(delta['content'])
#     return collected_messages
#     # except:
#     #     return ['Technical difficulties, please come back another time.']

# def generate_messages(messages: list) -> list:
#     formatted_messages = [
#         {
#             'role': 'system',
#             'content': 'You are a funny joke bot'
#         }
#     ]
#     for m in messages:
#         formatted_messages.append({
#             'role': 'user',
#             'content': m[0]
#         })
#         if m[1]!= None:
#             formatted_messages.append({
#                 'role': 'assistant',
#                 'content': m[1]
#             })
#     return formatted_messages


# def generate_response(chat_history: list) -> list:
#     messages = generate_messages(chat_history)
#     bot_message = chat_completion(messages)
#     print(bot_message)
#     chat_history[-1][1] = ''
#     for bm in bot_message:
#         chat_history[-1][1] += bm
#         yield chat_history


# def set_user_response(user_message:str, chat_history:list) -> tuple:
#     chat_history += [[user_message, None]]
#     return '', chat_history

