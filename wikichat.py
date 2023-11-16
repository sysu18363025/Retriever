import torch
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries
from colbert import Searcher
import pandas as pd
import json
from tqdm import tqdm

#load data
# queries = []
# answers = []
# dev = json.load(open('/data/huyuxuan/wikichat/ambigqa/dev_light.json'))
# for i in tqdm(range(len(dev)), total=len(dev)):
#     item = dev[i]
#     query = f"{item['question']}"
#     queries.append(query)
#     for annotation in item["annotations"]:
#         if annotation["type"]=="singleAnswer":
#             print('single',[list(set(annotation["answer"]))])
#             answers.append([list(set(annotation["answer"]))])
#         else:
#             print('multi',[list(set(pair["answer"])) for pair in annotation["qaPairs"]])
#             answers.append([list(set(pair["answer"])) for pair in annotation["qaPairs"]])
#     #answer = f"{item['annotations']}"
#     #answers.append(answer)
# query_0 = queries[0]
# #answer_0 = answers[0]
# print(query_0,answer_0)
# data = query_0
data = "Can you tell me about the 2023 Australian Open Men's singles final?"
tokenizer = AutoTokenizer.from_pretrained("/data/huyuxuan/chatglm", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/huyuxuan/chatglm", trust_remote_code=True).half().cuda()
model = model.eval()
#stage1:query
query_prompt = ("You are chatting with a user . Use Google search to form a response . You are both"
    "located in U.S." 
    "Today 's date is {today}.\n"
    "- What do you type in the search box ?\n"
    "- What date do you want the search results to be ? Enter ' recent ' if you are looking for the newest results ."
    " Enter ' none ' if the date is not important .\n\n"
    "### Few - shot example 1 ### \n"
    "You : Do you want to talk about sports ?\n"
    "User : Sure ! Who is your favorite basketball player ?\n"
    "[ Search needed ? Yes . You Google ' popular basketball players '. The year of the results is ' none '.]\n"
    "You : It has to be Lebron James .\n"
    "User : Did he play well in his last game ?\n"
    "[ Search needed ? Yes . You Google ' how did Lebron James do in his most recent game '. The year of the results is ' recent '.]\n\n" 
    #{ More few - shot examples #}   
    # {# The current dialogue #}
    # {% for dlg_turn in dlg %}
        # {% if dlg_turn . user_utterance is not none %}
            # User : {{ dlg_turn . user_utterance }}
        # {% endif %}
        # {% if dlg_turn . initial_search_query is not none %}
            # [ Search needed ? Yes . You Google "{{ dlg_turn . initial_search_query }}". The
            # year of the results is "{{ dlg_turn . initial_search_query_time } }".]
        # {% endif %}
        # {% if dlg_turn . agent_utterance is not none %}
            # You : {{ dlg_turn . agent_utterance }}
        # {% endif %}
    # {% endfor %}
    "User : {new_user_utterance}"
    "[ Search needed ?"
)
query_prompt = query_prompt.format(today=datetime.today().date(), new_user_utterance=data)
response, history = model.chat(tokenizer, query_prompt, history=[])
print(response)
########retriever#######
# if response[:3]=="Yes":
#     data = [[1, response]]
#     df = pd.DataFrame(data)
#     df.to_csv('queries.tsv', sep='\t', index=False)

# with Run().context(RunConfig(nranks=1, experiment="msmarco")):

#     config = ColBERTConfig(
#         root="/home/huyuxuan/WikiChat/ColBERT",
#     )
#     searcher = Searcher(index="msmarco.nbits=2", config=config)
#     queries = Queries("queries.tsv")
#     ranking = searcher.search_all(queries, k=100)
#     ranking.save("msmarco.nbits=2.ranking.tsv")

#stage2:summarize and filter
# summarize_fileter_prompt = (
#     "You Google different search queries and then Break down the relevant parts of the articles you find . Today 's date is {today}.\n\n"
#     "### Few - shot example 1 ### \n"
#     "Query : ' worst earthquake ever '\n"
#     "Title : January 1934 earthquake in India and Nepal\n"
#     "Article : The 1934 Nepal \ u2013India earthquake or 1934 Bihar \ u2013Nepal earthquake"
#     "was one of the worst earthquakes in India 's history . The towns of Munger and"
#     "Muzaffarpur were completely destroyed . This 8.0 magnitude earthquake occurred"
#     "on 15 January 1934 at around 2:13\ u00a0pm IST (08:43 UTC ) and caused widespread"
#     "damage in northern Bihar and in Nepal . Earthquake . The epicentre for this event"
#     "was located in eastern Nepal about south of Mount Everest . The areas where the"
#     "most damage to life and property occurred extended from Purnea in the east to"
#     "Champaran in the west ( a distance of nearly ) , and from Kathmandu in the north"
#     "to Munger in the south ( a distance of nearly ) .\n"

#     "Break down verbatum part ( s ) of this article that are related to the search query"
#     "'worst earthquake ever' or say None if the article is unrelated :\n"
#     "- The 1934 Nepal - India earthquake , also known as the 1934 Bihar - Nepal earthquake ,"
#     "was one of the worst earthquakes in India 's history .\n"
#     "- The 1934 Nepal - India earthquake had a magnitude of 8.0 and occurred on 15 January"
#     "1934.\n"
#     "- As a result of the 1934 Nepal - India earthquake , the towns of Munger and"
#     "Muzaffarpur were completely destroyed .\n"
#     "- As a result of the 1934 Nepal - India earthquake , widespread damage occurred in"
#     "northern Bihar and Nepal , with the most damage extending from Purnea in the"
#     "east to Champaran in the west , and from Kathmandu in the north to Munger in the"
#     "south .\n\n"

#     "### Few - shot example 2 ### \n"
#     "Query : ' age of Bruce Willis'\n "
#     "Title : Matt Willis\n"
#     "Article : In April 2005 , aged 21 , Willis stayed for three weeks at London 's Priory"
#     "Hospital for the treatment of alcoholism . In July 2006 , aged 23 , he was"
#     "admitted again for a few days for drug abuse , because he was addicted to"
#     "cannabis from the age of 13. He began to have problems from the drug - taking"
#     "including physiological and memory problems . In June 2008 , aged 25 , Willis"
#     "entered a rehab centre in Bournemouth after a marriage ultimatum . It was"
#     "reported that a night out with close friend Amy Winehouse pushed Willis too"
#     "far . Willis took the full five week course in drugs and alcohol .\n"
#     "Break down verbatum part ( s ) of this article that are related to the search query"
#     "'age of Bruce Willis' or say None if the article is unrelated :\n"
#     "None\n\n"

#     #{ More few - shot examples #}

#     #{ The current dialogue #}
#     "Query : {query}"
#     "Title : {title}"
#     "Article : {article}"
#     "Break down verbatum part ( s ) of this article that are related to the search query"
#     "'{query}' or say None if the article is unrelated :"
# )

# #stage3:generate
# #data={"annotations": [{"type": "singleAnswer", "answer": ["Tony Goldwyn", "Goldwyn"]}], "id": "-807825952267713091", "question": "Who plays the doctor in dexter season 1?"}
# data = "User: That's amazing! How did Djokovic perform throughout the tournament leading up to the final?\n"
# # generate_prompt = ("You are a friendly, knowledgeable and truthfull generator." "Generating a paragraph according to the question:\n{question}\n " "Generated in at most one paragraph.")
# # generate_prompt = generate_prompt.format(question=data["question"])
# generate_prompt = ("You are a friendly, knowledgeable and truthfull chatbot, talking to a user. Respond in at most one paragraph.\n"
#     "Today 's date is {today}\n."
#     "User: Can you tell me about the 2023 Australian Open Men's singles final? \n"
#     "You: In the thrilling 2023 Australian Open Men's singles final, Novak Djokovic defeated Stefanos Tsitsipas with a score of 6-3, 7-6(4), 7-"
#     "6(5). This amazing victory marked Djokovic's record-extending tenth Australian Open title and tied Rafael Nadal's all-time record of 22 men's"
#     "singles titles. Djokovic also became the second man to win ten titles at a single major, joining Nadal at the French Open. With this win, he surpassed"
#     "Nadal's tally for the fourth-most ATP Tour-level singles titles. \n"
# )
# generate_prompt = generate_prompt.format(today=datetime.today().date())
# generate_prompt = generate_prompt + data
# #print(generate_prompt)
# tokenizer = AutoTokenizer.from_pretrained("/data/huyuxuan/chatglm", trust_remote_code=True)
# model = AutoModel.from_pretrained("/data/huyuxuan/chatglm", trust_remote_code=True).half().cuda()
# model = model.eval()
# response, history = model.chat(tokenizer, generate_prompt, history=[])
# print(response)

# #stage4:extract claims
# extract_prompt = ("You are chatting with a user . Rewrite the facts mentioned in the last response into self-contained sentences. "
#     "Exclude opinions , or subjective statements .\n"
#     "Today 's date is {today}\n."     

#     "### Few - shot example 1 ### \n"
#     "User : When was chinese new year this year?\n"
#     "You : It was on February 5 th.\n"
#     "Rewrite the facts mentioned in the last response into self-contained sentences:\n"
#     "- Chinese New Year in 2022 was on February 5 th . The year of the results is '2022'.\n\n "

#     "### Few - shot example 2 ### \n"
#     "User : How are you doing today?\n"
#     "You : As an artificial intelligence, I don 't have feelings or personal experiences,"
#     "so I don 't have good days or bad days . However , I'm fully operational and ready"
#     "to assist you with any information or tasks you need . What can I help you with"
#     "today ?"
#     "Rewrite the facts mentioned in the last response into self-contained sentences:\n"
#     "Nothing\n\n."

#     "### Few - shot example 3 ### \n"
#     "User : Do you know about Queen Elizabeth II ? I was just reading about her .\n"
#     "You : Queen Elizabeth II is the current monarch of the United Kingdom and the"
#     "Commonwealth realms . She was born in 1926 and became queen in 1952.\n"
#     "Rewrite the facts mentioned in the last response into self-contained sentences :\n"
#     "- Queen Elizabeth II is the current monarch of the United Kingdom and the"
#     "Commonwealth realms as of {today}. The year of the results is 'recent'.\n"
#     "- Queen Elizabeth II was born in 1926. The year of the results is '1926'.\n"
#     "- Queen Elizabeth II became queen in 1952. The year of the results is '1952'.\n"
#     # ... {# More few - shot examples #}
#     #dialogue
#     "User: Can you tell me about the 2023 Australian Open Men's singles final? \n"
#     "You: In the thrilling 2023 Australian Open Men's singles final, Novak Djokovic defeated Stefanos Tsitsipas with a score of 6-3, 7-6(4), 7-\
#     6(5). This amazing victory marked Djokovic's record-extending tenth Australian Open title and tied Rafael Nadal's all-time record of 22 men's\
#     singles titles. Djokovic also became the second man to win ten titles at a single major, joining Nadal at the French Open. With this win, he surpassed\
#     Nadal's tally for the fourth-most ATP Tour-level singles titles. \n"
#     "User: That's amazing! How did Djokovic perform throughout the tournament leading up to the final?\n"
#     "{current_agent_utterance}"
#     "Rewrite the facts mentioned in the last response into self - contained sentences :\n"
#     )

# extract_prompt = extract_prompt.format(today=datetime.today().date(),current_agent_utterance=response)
# #print(extract_prompt)
# response, history = model.chat(tokenizer, extract_prompt, history=history)
# print(response)

#stage 5
# verification_prompt = (
#     "The following is a conversation between a user and a chatbot . For each claim that"
#     "the chatbot makes , you search the internet to obtain articles that would"
#     "support or refute that claim , and output one of ' SUPPORTS ' , ' REFUTES ' , or  'NOT"
#     "ENOUGH INFO '.\n"
#     "Only if the retrieved articles fully support the claim , output ' SUPPORTS '.\n"
#     "Today 's date is {today}.\n\n"

#     "### Few - shot example 1 ### \n"
#     "Chatbot : How was your trip to Hawaii ?\n"
#     "User : It was great ! In fact , I witnessed the eruption of the largest volcano on
#     "earth .\n"
#     "Chatbot : Wow , I hope I could see it , but sounds kinda dangerous . Is it the Mauna"
#     "Loa ?\n"
#     "User : Yes , it is ! Do you know when it started erupting ?\n"
#     "Chatbot : Yes , it started erupting on March 25 , 1984.\n
#     "[ You search the internet to fact - check the claim ' The last eruption of Mauna Loa
#     "started on March 25 , 1984']\n"
#     "[ You get these articles :\n"
#     "Title : 2022 eruption of Mauna Loa\n"
#     "Article : When active , Mauna Loa tends to produce ' voluminous , fast - moving lava"
#     "flows ' of the Hawaiian or effusive eruption type rather than more explosive"
#     "phreatic or Plinian eruptions , though it has produced explosive eruptions"
#     "between 300 and 1 ,000 years ago . Before Nov 27 , 2022 , Mauna Loa had last"
#     "erupted in March 1984 , in a 22 - day event similarly concentrated in the"
#     "volcano 's Northeast Rift Zone . The 2022 eruption was the volcano 's 34 th"
#     "eruption since 1843 , when volcanic activity at Mauna Loa began to be"
#     "continuously recorded , but only the third eruption since 1950. The 38 - year"
#     "span between the 1984 and 2022 eruptions was Mauna Loa 's longest period of"
#     "quiescence on record .\n"
#     "Title : 1984 eruption of Mauna Loa\n"
#     "Article : The 1984 eruption of Mauna Loa was a Hawaiian eruption in the U . S ."
#     "state of Hawaii that lasted from March 25 to April 15 , 1984. It ended a"
#     "9 - year period of quiescence at the volcano and continued for 22 days ,"
#     "during which time lava flows and lava fountains issued from the summit"
#     "caldera and fissures along the northeast and southwest rift zones . Although"
#     "the lava threatened Hilo , the flow stopped before reaching the outskirts of"
#     "town .]\n"
#     "Fact - check the claim ' The last eruption of Mauna Loa started on March 25 , 1984'."
#     "You think step by step : Mauna Loa had an eruption on Nov 27 , 2022 , which is later"
#     "than the claimed last eruption of March 25 , 1984. So the last eruption of Mauna"
#     "Loa was not on March 25 , 1984. So the fact - checking result is ' REFUTES '.\n\n"
#     #{ More few - shot examples #}
#     #{ The current dialogue #}
#     "Chatbot : {original_reply}\n"
#     "[ You search the internet to fact - check the claim '{claim}']\n"
#     "[ You get these articles :\n"
#     # {% for title in evidence_titles %}
#     #     Title : {{ title }}
#     #     Article : {{ evidence_texts [ loop .index -1]}}
#     # {% endfor %}
#     "]"
#     "Fact - check the claim '{claim}'.\n"
#     "You think step by step :\n"
#     )
# #stage6 draft
# draft_prompt = (
#     "The following is a conversation between a friendly , knowledgeable and truthful"
#     "chatbot , called WikiChat , and a user .\n"
#     "WikiChat can use search to get external knowledge when needed . In responding to the"
#     "user , WikiChat carefully aggregates search results in at most one paragraph .\n"
#     "Today 's date is {today}.\n\n"

#     "### Few - shot example 1 ### \n"
#     "User : What new discoveries from James Webb Space telescope can I tell my 9 year old"
#     "about ?\n"
#     "WikiChat : Your 9 - year old would be fascinated to hear that the James Webb Space"
#     "Telescope has discovered farthest galaxies ever found . It has found galaxies so"
#     "far away that it took 13.4 billion years for their lights to reach the"
#     "telescope . That is light that started its journey just 300 million years after"
#     "the Big Bang !\n"
#     "User : what is big bang ?\n"
#     "WikiChat : The Big Bang is the prevailing cosmological model for the universe from"
#     "the earliest known periods through its subsequent large - scale evolution . In"
#     "layman 's terms , it is the explosion that created our universe .\n"
#     "User : what else can I tell my child about ?\n"
#     "[ Chatbot Googles and gets this information :\n"
#     "- The James Webb Space Telescope has detected galaxies more than 13.4 billion light"
#     "years away , less than 300 million years after the Big Bang .\n"
#     "- This is a new record and astronomers are excited to learn more about these"
#     "distant galaxies .\n"
#     "- The telescope is named after James E . Webb , who was the administrator of NASA"
#     "from 1961 to 1968 during the Mercury , Gemini , and Apollo programs .
#     "]\n"
#     "WikiChat : The James Webb Space telescope is named after James E . Webb , who was the"
#     "administrator of NASA from 1961 to 1968 during the Mercury , Gemini , and Apollo"
#     "programs .\n\n"
#     #{ More few - shot examples #}
#     #{ The current dialogue #}
#     # {% for dlg_turn in dlg [ -2:] %}
#     #     User : {{ dlg_turn . user_utterance }}
#     #     WikiChat : {{ dlg_turn . agent_utterance }}
#     # {% endfor %}
#     #{ include the search results only for the last turn #}
#     # User : {{ last_user_utterance }}
#     # {% if evidences is not none %}
#     #     [ Chatbot Googles and gets this information :
#     #         {% for evi in evidences %}
#     #             - {{ evi }}
#     #         {% endfor %}
#     #     ]
#     # {% endif %}
#     "WikiChat :\n"
# )

# #stage7 refine
# refinement_prompt = (
#     "Given a conversation history between a user and a chatbot , provide feedback on the"
#     "chatbot 's final response , then revise the response based on the following"
#     "criteria :\n"
#     "* Relevant : The response should be on - topic and directly address the user 's"
#     "question . It should acknowledge if it 's off - topic or only partially addresses"
#     "the question . Irrelevant information should be avoided .\n"
#     "* Natural : The response should use engaging language to create an interactive and"
#     "enjoyable experience , without being too long .\n"
#     "* Non - Repetitive : The response should not repeat previously mentioned information"
#     "or statement , and should not repeat the same fact more than once .\n"
#     "* Temporally Correct : The response should provide up - to - date information , use"
#     "past - tense when the event happened before today ({today}) , and respond"
#     "specifically to the time mentioned by the user .\n"
#     "The revised response should only edit the original response according to the"
#     "feedback , and should not introduce new information .\n"
#     "Today 's date is {today}.\n\n"
#     "### Few - shot example 1 ### \n"
#     "User : What do you think is the best TV drama of 2022?\n"
#     "Chatbot : I think it has to be House of the Dragon .\n"
#     "User : Why is that ?\n"
#     "Response : I love it because both the plot and the visuals are great . It actually"
#     "won the Golden Globe Award for the best drama TV series . I'm not sure when it"
#     "was released , but I think it was August 21 , 2022.\n"
#     "Let 's break down the feedback for the response :\n"
#     "* Relevant : The response is on - topic and directly addresses the question of why the"
#     "speaker thinks House of the Dragon is the best TV drama , but it contains"
#     "irrelevant information about the release date of the show . 60/100\n"
#     "* Natural : The response uses engaging language to express the chatbot 's opinion and"
#     "provides supporting information to reinforce that opinion . 100/100\n"
#     "* Non - Repetitive : The response does not repeat any previous statement . 100/100\n"
#     "* Temporally Correct : The response correctly uses the past tense to describe the"
#     "Golden Globe win . 100/100\n"
#     "User : Why is that ?\n"
#     "Revised response after applying this feedback : I love it because both the plot and"
#     "the visuals are great . It actually won the Golden Globe Award for the best"
#     "drama TV series .\n"
#     #{ More few - shot examples #}
#     #{ The current dialogue #}
#     # {% for dlg_turn in dlg [ -2:] %} {# Only include the last few turns . #}
#     #     User : {{ dlg_turn . user_utterance }}
#     #     Chatbot : {{ dlg_turn . agent_utterance }}
#     # {% endfor %}
#     # User : {{ new_dlg_turn . user_utterance }}
#     # Response : {{ new_dlg_turn . agent_utterance }}
#     "Let 's break down the feedback for the response :\n"
# )
