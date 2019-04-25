python topic_mining.py \
	--train_file "/data/xuht/free_text_topmines/20190424/mining_qince.txt&/data/xuht/free_text_topmines/mining_source.txt" \
	--stop_word_file "/data/xuht/stopwords-zh-master/stopwords-zh.txt" \
	--ouput_file "/data/xuht/free_text_topmines/20190424/global_mining_unigram" \
	--num_topics 2 \
	--iteration 1000 \
	--optimization_burnin 100 \
	--alpha 4 \
	--optimization_iterations 50 \
	--beta 0.02 \
	--min_support 10 \
	--max_phrase_size 5

