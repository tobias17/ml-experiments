
CMD = python 15_cheat_sheet/train.py

test:
	$(CMD)

run:
	JITBEAM=20 $(CMD)
