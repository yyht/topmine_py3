from topmine_src import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
	"corpus", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_prefix", None,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_integer(
	"vocab_size", 50000,
	"Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
	"model_type", "char",
	"class-related topic and class-unrelated topic")

flags.DEFINE_float(
	"character_coverage", 0.9995,
	"Input TF example files (can be a glob or comma separated).")

def main(_):
	config = {}
	config["input"] = FLAGS.corpus
	config["model_prefix"] = FLAGS.model_prefix
	config["vocab_size"] = FLAGS.vocab_size
	config["model_type"] = FLAGS.model_type
	config["character_coverage"] = FLAGS.character_coverage
	tokenization_api = tokenization.SPM(config)
	tokenization_api.train_model()


if __name__ == "__main__":
	tf.app.run()
