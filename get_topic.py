from collections import Counter
import _pickle as pkl
from topmine_src import utils
import tensorflow as tf
from fuzzywuzzy import fuzz
import re
import _pickle as pkl
import json

from collections import OrderedDict

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"mining_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"prediction_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"doc_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"vocab_path", None,
	"class-related topic and class-unrelated topic")

flags.DEFINE_string(
	"data_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"label_mapping_path", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_path", None,
	"Input TF example files (can be a glob or comma separated).")

def read_topic_model(path):
	with open(path, "rb") as frobj:
		mining_info = pkl.load(frobj)

	return mining_info

def get_frequent_phrases(mining_info):
	return mining_info["frequent_phrases"]

def get_topics(mining_info):
	return mining_info["topics"]

def get_index_word(mining_info):
	return mining_info["index_vocab"]

def get_id2label(data_path):
	with open(data_path, "r") as frobj:
		id2label = OrderedDict()
		for i, line in enumerate(frobj):
			content = line.strip().split()
			label = content[0].split("__label__")[1]
			id2label[i] = label

	return id2label

def get_most_frequent_topics(mining_info):
	topics = get_topics(mining_info)
	index_vocab = get_index_word(mining_info)
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

def parse_phrases(mining_info, prediction_info):
	data = mining_info["frequent_phrases"]
	pred = prediction_info["pred_label"]

	indexer = mining_info["indexer"]["partitioned_docs_indexer"]
	filtered = []

	for p, d in zip(pred, data):
		if p == 0:
			continue
		else:
			filtered.append(d)
	return filtered

def get_label_mapping(label_mapping_path):
	with open(label_mapping_path, "r") as frobj:
		mapping_dict = json.load(frobj)
		return mapping_dict

def get_indicator(mining_info, prediction_info, doc_path, vocab_path,
				data_path, label_mapping_path, output_path, **kargs):

	label_mapping = get_label_mapping(label_mapping_path)
	indexer = mining_info["indexer"]["partitioned_docs_indexer"]

	docs = utils.load_partitioned_docs(doc_path)
	index_word = utils.load_vocab(vocab_path)

	id2label = get_id2label(data_path)

	str_lst = []
	for i, doc in enumerate(docs):
		string = []
		for sub_word in doc:
			string.append((" ".join([index_word[index] for index in sub_word])))
		str_lst.append(string)

	phrases = parse_phrases(mining_info, prediction_info)
	print("==filtered phrases==", len(phrases))
	indicator_lst = []

	for s_index, s in enumerate(str_lst):
		indicator = [0]*len(s)
		pattern = []
		for index, sub_s in enumerate(s):
			if sub_s in phrases:
				indicator[index] = 1
				if len(pattern) >= 1:
					flag = 0
					for sub in pattern:
						score = fuzz.ratio(sub, sub_s)/ 100.0
						if score >= 0.9:
							flag = 1
					if flag == 0:
						pattern.append(sub_s)
				else:
					pattern.append(sub_s)

		pattern = [re.sub(" ", "", sub_pattern) for sub_pattern in pattern]
		doc_string = "".join("".join(s).split())

		if s_index == 0:
			print(doc_string)

		string_id = indexer.get(doc_string, "none")

		label_id = id2label.get(string_id, "0")
		cn_label = label_mapping[label_id]

		indicator_lst.append({"indicator":indicator,
							"segmented_sentence":s,
							 "pattern":"&".join(pattern),
							 "doc2string_index":string_id,
							 "label_id":label_id,
							 "cn_label":cn_label
							 })

	pkl.dump(indicator_lst, open(output_path, "wb"))

	return indicator_lst

def main(_):

	mining_info = pkl.load(open(FLAGS.mining_path, "rb"))
	prediction_info = pkl.load(open(FLAGS.prediction_path, "rb"))
	indicator_lst = get_indicator(mining_info, prediction_info, 
				FLAGS.doc_path, FLAGS.vocab_path, FLAGS.data_path,
				FLAGS.label_mapping_path, FLAGS.output_path)

if __name__ == "__main__":
	tf.app.run()


	