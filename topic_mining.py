from topmine_src import phrase_lda
from topmine_src import phrase_mining
from topmine_src import utils
import re, json
from hanziconv import HanziConv
from tqdm import tqdm
import numpy as np

import tensorflow as tf

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
	"ouput_file", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"num_topics", 2,
	"class-related topic and class-unrelated topic")

flags.DEFINE_integer(
	"iteration", 1000,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"optimization_burnin", 100,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"optimization_iterations", 50,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_float(
	"beta", 0.01,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"min_support", 10,
	"represents the minimum number of occurences you want each phrase to have..")

flags.DEFINE_integer(
	"alpha", 4,
	"represents the threshold for merging two words into a phrase. A lower value alpha leads to higher recall and lower precision,.")

flags.DEFINE_integer(
	"max_phrase_size", 10,
	"length of the maximum phrase size.")

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

def main(_):
	stop_word_file = FLAGS.stop_word_file

	file_name = FLAGS.train_file

	min_support = FLAGS.min_support
	max_phrase_size = FLAGS.max_phrase_size
	alpha = FLAGS.alpha
	beta = FLAGS.beta
	iteration = FLAGS.iteration
	num_topics = FLAGS.num_topics
	optimization_iterations = FLAGS.optimization_iterations
	optimization_burnin = FLAGS.optimization_burnin

	import jieba

	# with open(FLAGS.train_file, "r") as frobj:
	# 	examples = [line.strip() for line in frobj]
	# 	print(len(examples), "===before removing duplicate===")
	# 	examples = set(examples)
	# 	tmp = []
	# 	for example in examples:
	# 		re_pattern = "({}{})".format("__label__", "\d.")
	# 		element_list = re.split(re_pattern, example)
	# 		tmp.append(" ".join(list(jieba.cut("".join(element_list[-1].split())))))
	# 	examples = set(tmp)
	# 	print(len(examples), "===after removing duplicate===")

	train_file_list = FLAGS.train_file.split("&")
	examples = []
	for train_file in train_file_list:
		with open(train_file, "r") as frobj:
			for line in tqdm(frobj):
				try:
					content = json.loads(line)
					text = "".join(content["text"].split())
					examples.append(" ".join(list(jieba.cut(clean(text)))))
				except:
					continue

	def _get_stopwords(stop_word_path):
		"""
		Returns a list of stopwords.
		"""
		stopwords = set()
		with open(stop_word_path, "r") as frobj:
			for line in frobj:
				stopwords.add(line.rstrip())
		return stopwords

	stopwords = _get_stopwords(FLAGS.stop_word_file)

	phrase_miner = phrase_mining.PhraseMining(min_support, max_phrase_size, alpha)
	partitioned_docs, index_vocab, partitioned_indexer = phrase_miner.mine(examples, stopwords)
	frequent_phrases = phrase_miner.get_frequent_phrases(min_support, if_only_phrase=False)
	partioned_docs_path = FLAGS.ouput_file + "/partioned_docs.txt"
	utils.store_partitioned_docs(partitioned_docs,
								 path=partioned_docs_path)
	vocab_path = FLAGS.ouput_file + "/vocabs.txt"
	utils.store_vocab(index_vocab, path=vocab_path)

	frequent_phrase_path = FLAGS.ouput_file + "/frequent_phrases.txt"
	utils.store_frequent_phrases(frequent_phrases,
								 path=frequent_phrase_path)
	print("{}: total frequent phrases {}".format(file_name, len(frequent_phrases)))

	# print('Running PhraseLDA...')

	# partitioned_docs = utils.load_partitioned_docs(path=partioned_docs_path)
	# vocab_file = utils.load_vocab(path=vocab_path)

	# plda = phrase_lda.PhraseLDA( partitioned_docs, vocab_file, num_topics ,
	# 			alpha, beta, iteration, optimization_iterations, optimization_burnin);

	# document_phrase_topics, most_frequent_topics, topics = plda.run()

	# stored_topics_path = FLAGS.ouput_file + "/doc_phrase_topics.txt"
	# utils.store_phrase_topics(document_phrase_topics,
	# 						 path=stored_topics_path)
	# most_frequent_topic_prefix_path = FLAGS.ouput_file + "/frequent_phrase_topics.txt"
	# utils.store_most_frequent_topics(most_frequent_topics,
	# 								prefix_path=most_frequent_topic_prefix_path)

	import _pickle as pkl
	pkl.dump({"frequent_phrases":frequent_phrases,
		"index_vocab":index_vocab,
		"partitioned_docs":partitioned_docs,
		"indexer":partitioned_indexer},
		open(FLAGS.ouput_file+"/mining_info.pkl", "wb"))
	# pkl.dump({"frequent_phrases":frequent_phrases,
	# 	"index_vocab":index_vocab,
	# 	"partitioned_docs":partitioned_docs,
	# 	"document_phrase_topics":document_phrase_topics,
	# 	"topics":topics,
	# 	"most_frequent_topics":most_frequent_topics},
	# 	open(FLAGS.ouput_file+"/mining_info.pkl", "wb"))

if __name__ == "__main__":
	tf.app.run()
