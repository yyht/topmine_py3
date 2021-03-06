python topic_mining_pkl.py \
	--train_file "/data/xuht/topmine/free_text/free_text.pkl" \
	--stop_word_file "/data/xuht/stopwords-zh-master/stopwords-zh.txt" \
	--ouput_file "/data/xuht/topmine/free_text/" \
	--num_topics 2 \
	--iteration 1000 \
	--optimization_burnin 100 \
	--alpha 4 \
	--optimization_iterations 50 \
	--beta 0.02 \
	--min_support 5 \
	--max_phrase_size 10 \
	--if_only_phrase True
