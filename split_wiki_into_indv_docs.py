import os
import jsonlines
import codecs
import json
import datetime

wiki_folder = 'data/wiki-pages'
dest_dir = "data/wiki-pages-split"
files = os.listdir(wiki_folder)

count = 0
emptyPagesCounter= 0
numLines = 0
emptyLine = 0
numEntities = 0
emptyEntities = 0

for file in files:
	fileContent = jsonlines.open(wiki_folder + "/" + file)

	for page in fileContent:
		page['id'] = page['id'].replace("/", "-SLH-")
		page['id'] = page['id'].encode('utf8').decode('utf8')
		# print(page['id'])

		# new_file = codecs.open(dest_dir + "/" + page['id'] + ".txt", "w+","utf-8")

		# preprocessing lines
		doc_splitted_lines = page["lines"].split("\n")

		if len(page["text"]) > 0:
			count = count + 1
			for line in doc_splitted_lines:
				numLines = numLines + 1

				# sentences are organized as follows:
				#	SENTENCE_ID\t SENTENCE_TEXT\t NAMED_ENTITY1\t NAMED_ENTITY2

				splittedSentence = line.split("\t")
				if len(splittedSentence) >= 3:
					# linesList.append({"content": splittedSentence[1], "namedEntitiesList": splittedSentence[2:]})
					numEntities = numEntities + len(splittedSentence)
				elif len(splittedSentence) == 2:
					# linesList.append({"content": splittedSentence[1], "namedEntitiesList": []})
					numEntities = numEntities + len(splittedSentence)
				else:
					# TODO: this happened at least one time -> this happens when a line does not contain an id and does not contain the symbol "\t". What should be done? For now, ignore it!
					print("[WARNING] Article " + page['id'] + " found on the file " + str(
						file) + "contains text without id!")
					emptyEntities = emptyEntities + 1
					continue
		else:
			print("[WARNING] Article " + page['id'] + " is empty!")
			emptyPagesCounter = emptyPagesCounter + 1

print("Parsed successfully")
print(datetime.datetime.now())
print("number of empty articles 	= \t" + str(emptyPagesCounter))
print("number of files 				= \t" + str(count))
print("number of lines 				= \t" + str(numLines))
print("number of entities 			= \t" + str(numEntities))
print("number of articles w/out id	= \t" + str(emptyEntities))