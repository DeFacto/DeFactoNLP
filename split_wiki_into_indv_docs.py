import os
import jsonlines
import codecs
import json
import datetime

wiki_folder = 'data/wiki-pages/wiki-pages'
dest_dir = "data/wiki-pages/wiki-pages-split"
files = os.listdir(wiki_folder)


for file in files:
	fileContent = jsonlines.open(wiki_folder + "/" + file)
	
	emptyPagesCounter= 0
	
	for page in fileContent:
		page['id'] = page['id'].replace("/","-SLH-")
		page['id'] = page['id'].encode('utf8').decode('utf8')
		#print(page['id'])
		
		#new_file = codecs.open(dest_dir + "/" + page['id'] + ".txt", "w+","utf-8")
		
		"""
		lines = page['lines'].split("\n")
		for i in range(len(lines)):
			lines[i] = lines[i].strip()
			lines[i] = lines[i].replace("\t"," ")
			lines[i] = lines[i][2:]
			new_file.write(lines[i])
			if i != len(lines)-1:
				new_file.write("\n")
		"""
		
		# preprocessing lines
		doc_splitted_lines= page["lines"].split("\n")
		
		if len(page["text"]) > 0:
			
			new_file = codecs.open(dest_dir + "/" + page['id'] + ".json", "w+","utf-8")
			
			linesList= []
			
			for line in doc_splitted_lines:
				# sentences are organized as follows:
				#	SENTENCE_ID\tSENTENCE_TEXT\tNAMED_ENTITY1\tNAMED_ENTITY2
				splittedSentence= line.split("\t")
				
				if len(splittedSentence) >= 3:
					linesList.append({"content": splittedSentence[1], "namedEntitiesList": splittedSentence[2:]})
				elif len(splittedSentence) == 2:
					linesList.append({"content": splittedSentence[1], "namedEntitiesList": []})
				else:
					#TODO: this happened at least one time -> this happens when a line does not contain an id and does not contain the symbol "\t". What should be done? For now, ignore it!
					print("[WARNING] Article " + page['id'] + " found on the file " + str(file) + "contains text without id!")
					continue
				
			
			json.dump({"text": page['text'], "lines": linesList}, new_file)
			new_file.close()
		else:
			print("[WARNING] Article " + page['id'] + " is empty!")
			emptyPagesCounter = emptyPagesCounter + 1
			
	
	print("Parsed successfully")
	print(datetime.datetime.now())
	print("number of empty articles= " + str(emptyPagesCounter))

