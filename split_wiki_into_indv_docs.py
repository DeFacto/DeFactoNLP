import os
import jsonlines
import codecs
import json

wiki_folder = 'data/wiki-pages/wiki-pages'
dest_dir = "data/wiki-pages/wiki-pages-split"
files = os.listdir(wiki_folder)

for file in files:
	file = jsonlines.open(wiki_folder + "/" + file)
	for page in file:
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
		
		if len(doc_splitted_lines) > 1:
			
			new_file = codecs.open(dest_dir + "/" + page['id'] + ".json", "w+","utf-8")
			
			linesList= []
			
			for line in doc_splitted_lines:
				# sentences are organized as follows:
				#	SENTENCE_ID\tSENTENCE_TEXT\tNAMED_ENTITY1\tNAMED_ENTITY2
				splittedSentence= line.split("\t")
				
				if len(splittedSentence) >= 3:
					linesList.append({"content": splittedSentence[1], "namedEntitiesList": splittedSentence[2:]})
				else:
					linesList.append({"content": splittedSentence[1], "namedEntitiesList": []})
					
				
			
			json.dump({"text": page['text'], "lines": linesList}, new_file)
			new_file.close()
		else:
			print("[WARNING] Article " + page['id'] + " is empty!")
