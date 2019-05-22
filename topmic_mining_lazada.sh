python topic_mining_lazada.py \
	--train_file "/data/xuht/lazada/new_data/20190512/total_data.txt" \
	--stop_word_file "/data/xuht/stopwords-zh-master/stopwords-zh.txt" \
	--ouput_file "/data/xuht/lazada/new_data/20190512/global_mining_unigram" \
	--num_topics 2 \
	--iteration 1000 \
	--optimization_burnin 100 \
	--alpha 4 \
	--optimization_iterations 50 \
	--beta 0.02 \
	--min_support 10 \
	--max_phrase_size 5

