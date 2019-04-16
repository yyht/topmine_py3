from topmine_src import utils
import re, json

import numpy as np

import tensorflow as tf
from flash_text import KeywordProcessor

import _pickle as pkl

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"train_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"stop_word_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"mining_info", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"output_file", None,
	"class-related topic and class-unrelated topic")

def main(_):

	stop_word_file = FLAGS.stop_word_file

	file_name = FLAGS.train_file

	import jieba
	with open(FLAGS.train_file, "r") as frobj:
		examples = []
		for line in frobj:
			content = json.loads(line.strip())
			examples.append(content)

	with open(FLAGS.mining_info, "rb") as frobj:
		result = pkl.load(frobj)

	mined_phrases = result["frequent_phrases"]
	vocab_index = result["index_vocab"]
	partioned_docs = result["partitioned_docs"]
	doc_index = result["indexer"]

	keyword_detector = KeywordProcessor()

	vocab2id, id2vocab = {}, {}
	for index, word in enumerate(vocab_index):
		vocab2id[word] = index
		id2vocab[index] = word

	phrase_count = {}
	for item in mined_phrases:
		phrase_count[item[0]] = {}
		phrase_count[item[0]]["count"] = item[1]
		phrase_count[item[0]]["label"] = []
		keyword_detector.add_keyword(list("".join(item[0].split())), [item[0]])

	for index, example in zip(doc_index, partioned_docs):
		tmp_docs = []
		for phrase_id_lst in example:
			phrase_string = "".join([id2vocab[i] for i in phrase_id_lst])
			tmp_docs.append(phrase_string)
		tmp_docs = "".join(tmp_docs)
		output = keyword_detector.extract_keywords(list(tmp_docs), span_info=True)
		word_lst = [item[0][0] for item in output]
		for word in word_lst:
			if word in phrase_count:
				phrase_count[word]["label"].append(examples[index]["label"])
		
	with open(FLAGS.output_file, "wb") as fwobj:
		pkl.dump(phrase_count, fwobj)
