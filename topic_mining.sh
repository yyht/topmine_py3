python topic_mining.py \
	--train_file "/data/xuht/politics/train.txt" \
	--stop_word_file "/data/xuht/stopwords-zh-master/stopwords-zh.txt" \
	--ouput_file "/data/xuht/politics/global_mining" \
	--num_topics 2 \
	--iteration 1000 \
	--optimization_burnin 100 \
	--alpha 4 \
	--optimization_iterations 50 \
	--beta 0.02 \
	--min_support 10 \
	--max_phrase_size 10
