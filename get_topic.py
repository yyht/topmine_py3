from collections import Counter
import _pickle as pkl
from topmine_scr import utils

def read_topic_model(path):
	with open(path, "rb") as frobj:
		topic_info = pkl.load(frobj)

	return topic_info

def get_frequent_phrases(topic_info):
	return topic_info["frequent_phrases"]

def get_topics(topic_info):
	return topic_info["topics"]

def get_index_word(topic_info):
	return topic_info["index_vocab"]

def get_most_frequent_topics(topic_info):
	topics = get_topics(topic_info)
	get_index_word = get_index_word(topic_info)
	output = []
    topic_index = 0
    for topic in topics:
        output_for_topic = []
        print("topic", topic_index)
        for phrase, count in topic.most_common():
            if len(phrase.split(" ")) > 1:
                val = utils._get_string_phrase(phrase, index_vocab), count
                output_for_topic.append(val)
                print(val)
        output.append(output_for_topic)
        topic_index += 1
    return output

def postprocess_frequent_phrase(frequent_phrases):
	pass