from transformers import AutoTokenizer, AutoModel
from datetime import datetime
#stage3:generate
#data={"annotations": [{"type": "singleAnswer", "answer": ["Tony Goldwyn", "Goldwyn"]}], "id": "-807825952267713091", "question": "Who plays the doctor in dexter season 1?"}
data = "User: That's amazing! How did Djokovic perform throughout the tournament leading up to the final?\n"
# generate_prompt = ("You are a friendly, knowledgeable and truthfull generator." "Generating a paragraph according to the question:\n{question}\n " "Generated in at most one paragraph.")
# generate_prompt = generate_prompt.format(question=data["question"])
generate_prompt = ("You are a friendly, knowledgeable and truthfull chatbot, talking to a user. Respond in at most one paragraph.\n"
    "User: Can you tell me about the 2023 Australian Open Men's singles final? \n"
    "You: In the thrilling 2023 Australian Open Men's singles final, Novak Djokovic defeated Stefanos Tsitsipas with a score of 6-3, 7-6(4), 7-"
    "6(5). This amazing victory marked Djokovic's record-extending tenth Australian Open title and tied Rafael Nadal's all-time record of 22 men's"
    "singles titles. Djokovic also became the second man to win ten titles at a single major, joining Nadal at the French Open. With this win, he surpassed"
    "Nadal's tally for the fourth-most ATP Tour-level singles titles. \n"
)
generate_prompt = generate_prompt + data
#print(generate_prompt)
tokenizer = AutoTokenizer.from_pretrained("/data/huyuxuan/chatglm", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/huyuxuan/chatglm", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, generate_prompt, history=[])
print(response)

#stage4:extract claims
extract_prompt = ("You are chatting with a user . Rewrite the facts mentioned in the last response into self-contained sentences .\
    Exclude opinions , or subjective statements .\n"
    "Today 's date is {today}\n."     

    "### Few - shot example 1 ### \n"
    "User : When was chinese new year this year?\n"
    "You : It was on February 5 th.\n"
    "Rewrite the facts mentioned in the last response into self-contained sentences:\n"
    "- Chinese New Year in 2022 was on February 5 th . The year of the results is '2022'.\n\n "

    "### Few - shot example 2 ### \n"
    "User : How are you doing today?\n"
    "You : As an artificial intelligence, I don 't have feelings or personal experiences,"
    "so I don 't have good days or bad days . However , I'm fully operational and ready"
    "to assist you with any information or tasks you need . What can I help you with"
    "today ?"
    "Rewrite the facts mentioned in the last response into self-contained sentences:\n"
    "Nothing\n\n."

    "### Few - shot example 3 ### \n"
    "User : Do you know about Queen Elizabeth II ? I was just reading about her .\n"
    "You : Queen Elizabeth II is the current monarch of the United Kingdom and the"
    "Commonwealth realms . She was born in 1926 and became queen in 1952.\n"
    "Rewrite the facts mentioned in the last response into self-contained sentences :\n"
    "- Queen Elizabeth II is the current monarch of the United Kingdom and the"
    "Commonwealth realms as of {today}. The year of the results is 'recent'.\n"
    "- Queen Elizabeth II was born in 1926. The year of the results is '1926'.\n"
    "- Queen Elizabeth II became queen in 1952. The year of the results is '1952'.\n"
    # ... {# More few - shot examples #}
    #dialogue
    "User: Can you tell me about the 2023 Australian Open Men's singles final? \n"
    "You: In the thrilling 2023 Australian Open Men's singles final, Novak Djokovic defeated Stefanos Tsitsipas with a score of 6-3, 7-6(4), 7-\
    6(5). This amazing victory marked Djokovic's record-extending tenth Australian Open title and tied Rafael Nadal's all-time record of 22 men's\
    singles titles. Djokovic also became the second man to win ten titles at a single major, joining Nadal at the French Open. With this win, he surpassed\
    Nadal's tally for the fourth-most ATP Tour-level singles titles. \n"
    "User: That's amazing! How did Djokovic perform throughout the tournament leading up to the final?\n"
    "{current_agent_utterance}"
    "Rewrite the facts mentioned in the last response into self - contained sentences :\n"
    )

extract_prompt = extract_prompt.format(today=datetime.today().date(),current_agent_utterance=response)
#print(extract_prompt)
response, history = model.chat(tokenizer, extract_prompt, history=history)
print(response)
