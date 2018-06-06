import os
import jsonlines
import codecs

wiki_folder = 'data/wiki-pages/wiki-pages'
dest_dir = "data/wiki-pages/wiki-pages-split"
files = os.listdir(wiki_folder)

for file in files:
	file = jsonlines.open(wiki_folder + "/" + file)
	for page in file:
		page['id'] = page['id'].replace("/","-SLH-")
		print(page['id'])
		new_file = codecs.open(dest_dir + "/" + page['id'] + ".txt", "w+","utf-8")
		lines = page['lines'].split("\n")
		for i in range(len(lines)):
			lines[i] = lines[i].strip()
			lines[i] = lines[i].replace("\t"," ")
			lines[i] = lines[i][2:]
			new_file.write(lines[i])
			if i != len(lines)-1:
				new_file.write("\n")