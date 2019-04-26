from topmine_src import utils
import re, json

import numpy as np
import tensorflow as tf
from flash_text import KeywordProcessor
from collections import Counter
import _pickle as pkl
from topmine_src import phrase_mining
import jieba
from tqdm import tqdm


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

def clean(text):
	text = text.strip()
	text = re.sub(u"[\s\t\ue742◥ █ ◤]", "", text)
	text = re.sub(u"\n", u"。", text)
	text = re.sub(u"\n", "", text)
	text = re.sub(u"\\<.*?>", "", text)
	text = re.sub(u'&nbsp', "", text)
	text = re.sub(u"0a", "", text)
	text = re.sub(u"0 a", "", text)
	# text = re.sub(u"[^\u4e00-\u9fa5^.^a-z^A-Z^0-9{}]".format(CH_PUNCTUATION), "", text)
	return text

def main():

	def _get_stopwords(stop_word_path):
		"""
		Returns a list of stopwords.
		"""
		stopwords = set()
		with open(stop_word_path, "r") as frobj:
			for line in frobj:
				stopwords.add(line.rstrip())
		return stopwords

	stop_word_file = FLAGS.stop_word_file

	file_name = FLAGS.train_file
	stopwords = _get_stopwords(stop_word_file)

	# import jieba
	# with open(FLAGS.train_file, "r") as frobj:
	# 	examples = []
	# 	for line in frobj:
	# 		content = json.loads(line.strip())
	# 		examples.append(content)

	train_file_list = FLAGS.train_file.split("&")
	examples = []
	for train_file in train_file_list:
		with open(train_file, "r") as frobj:
			for line in tqdm(frobj):
				try:
					content = json.loads(line)
					content["text"] = clean("".join(content["text"].split()))
					examples.append(content)
				except:
					continue

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

	# unigram = {}
	# for item in mined_phrases:
	# 	unigram[item[0]] = {}
	# 	unigram[item[0]]["count"] = item[1]
	# 	unigram[item[0]]["label"] = []
	# 	unigram[item[0]]["ratio"] = []
	# 	keyword_detector.add_keyword(item[0].split(), [item[0]])

	# for index, example in zip(doc_index, partioned_docs):
	# 	for phrase_id_lst in example:
	# 		phrase_string = " ".join([id2vocab[i] for i in phrase_id_lst])
	# 		if phrase_string in unigram:
	# 			unigram[phrase_string]["label"].append(examples[index]["label"])

	# for word in unigram:
	# 	unigram[word]["ratio"] = Counter(unigram[word]["label"])

	# with open(FLAGS.output_file, "wb") as fwobj:
	# 	pkl.dump(unigram, fwobj)

# main()
	# phrase_count = {}
	for item in mined_phrases:
		# phrase_count[item[0]] = item[1]
		keyword_detector.add_keyword(item[0].split(), [item[0]])

	unigram = {}
	for index, example in enumerate(tqdm(examples)):
		content = list(jieba.cut(phrase_mining.clean(example["text"])))
		content = [word for word in content if word not in stopwords]
		output = keyword_detector.extract_keywords(content, span_info=True)
		word_lst = [item[0][0] for item in output]
		for word in word_lst:
			if word in unigram:
				unigram[word]["label"].append(example["label"])
				unigram[word]["count"] += 1
				unigram[word]["doc_id"].append(index)
			else:
				unigram[word] = {}
				unigram[word]["label"] = [example["label"]]
				unigram[word]["count"] = 1
				unigram[word]["doc_id"] = [index]

	print("==size of unigram==", len(unigram), len(mined_phrases))

	for word in unigram:
		unigram[word]["ratio"] = Counter(unigram[word]["label"])
		
	with open(FLAGS.output_file, "wb") as fwobj:
		pkl.dump(unigram, fwobj)

main()
