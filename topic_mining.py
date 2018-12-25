from topmine_src import phrase_lda
from topmine_src import phrase_mining
from topmine_src import utils

import numpy as np

import tensorflow as tf

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

	phrase_miner = phrase_mining.PhraseMining(file_name, min_support, max_phrase_size, alpha, stop_word_file)
	partitioned_docs, index_vocab = phrase_miner.mine()
	frequent_phrases = phrase_miner.get_frequent_phrases(min_support)
	partioned_docs_path = FLAGS.ouput_file + "/partioned_docs.txt"
	utils.store_partitioned_docs(partitioned_docs, 
								 path=partioned_docs_path)
	vocab_path = FLAGS.ouput_file + "/vocabs.txt"
	utils.store_vocab(index_vocab, path=vocab_path)

	frequent_phrase_path = FLAGS.ouput_file + "/frequent_phrases.txt"
	utils.store_frequent_phrases(frequent_phrases, 
								 path="frequent_phrase_path")
	print("{}: total frequent phrases {}".format(file_name, len(frequent_phrases)))
	
	print('Running PhraseLDA...')

	partitioned_docs = utils.load_partitioned_docs(path=partioned_docs_path)
	vocab_file = utils.load_vocab(path=vocab_path)

	plda = phrase_lda.PhraseLDA( partitioned_docs, vocab_file, num_topics , 
				alpha, beta, iteration, optimization_iterations, optimization_burnin);

	document_phrase_topics, most_frequent_topics = plda.run()

	stored_topics_path = FLAGS.ouput_file + "/doc_phrase_topics.txt"
	utils.store_phrase_topics(document_phrase_topics,
							 path=stored_topics_path)
	most_frequent_topic_prefix_path = FLAGS.ouput_file + "/frequent_phrase_topics.txt"
	utils.store_most_frequent_topics(most_frequent_topics,
									prefix_path=most_frequent_topic_prefix_path)

if __name__ == "__main__":
	tf.app.run()
