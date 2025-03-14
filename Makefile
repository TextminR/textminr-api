e:
	bpython src/example.py -i

de:
	scp -r src textminr:~/api
	scp .env textminr:~/api
	scp Makefile textminr:~/api

run:
	python -m fastapi dev src/main.py

dict:
	python src/dictionary.py

trimDict:
	python src/trimDictionary.py
