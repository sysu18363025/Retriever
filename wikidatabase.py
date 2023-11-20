import glob
import json
from blingfire import text_to_sentences
import pandas as pd
import re
wiki_dump_file_out='collection.tsv'
wiki_dump_folder_in='/data/huyuxuan/wikichat/wikidatabase2/AA/wiki_00'
#with open(wiki_dump_file_out, 'w', encoding='utf-8') as out_f:
database = []
for filename in glob.glob(wiki_dump_folder_in):
    filename=filename.replace("\\","/")
    articles = []
    for line in open(filename, 'r'):
        articles.append(json.loads(line))
        #print('articles',line)
    for article in articles:
        if article['text'] == "":
            database.append(article['title'])
            continue
        else:
            sentences = article['text'].split('. ')
            current_block = ''
            current_words_num = 0
            for sentence in sentences:
                sentence = re.compile('\n').sub(' ', sentence)
                words = sentence.split()
                current_sentence_words_num = len(words)
                if current_words_num + current_sentence_words_num + 1 <= 120:
                    if current_block:
                        current_block += " " + sentence 
                        current_words_num += current_sentence_words_num
                    else:
                        current_block = article['title'] + ': ' + sentence  
                        current_words_num = 1 + current_sentence_words_num                  
                else:
                    # 当前文本块已满，添加到文本块列表中，并重新开始
                    database.append(current_block)
                    # if current_sentence_words_num > 120:
                    #     current_block = article['title'] + ': ' + ' '.join(words[:120])
                    # #print(current_block)
                    # else:
                    current_block = article['title'] + ': ' + sentence
                    current_words_num = 1 + current_sentence_words_num
            if current_block:
                database.append(article['title'] + ': ' + sentence)
# data_index = []
# for i in range(len(database)):
#     data_index.append(i)
data_with_index = pd.Series(database)
print(data_with_index)
data_with_index.to_csv(wiki_dump_file_out, sep='\t', index=True)



            

