
CMD = python 15_cheat_sheet/train.py --restore /home/tiny/ml-experiments/weights/15_cheat_sheet/2025_07_06_17_43_30_857338

test:
	$(CMD)

run:
	JITBEAM=20 $(CMD)
