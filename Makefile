
CMD = python 13_token_cluster/train.py --restore /home/tiny/ml-experiments/weights/13_token_cluster/2024_10_18_18_51_56_467251 --predict-loss --decoded-loss

test:
	$(CMD)

beam:
	BEAM=5 $(CMD)
