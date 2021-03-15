import os

txt_path='./test_txt/'
with open('./test_labels.txt') as f:
	lines=f.readlines()
	for line in lines:
		line=line.strip()
		words=line.split('\t')
		file_name=words[0]
		img_size=words[1]
		coordinate=words[2]
		label=words[3]
		with open(txt_path+file_name[:-3]+'txt','a') as f1:
			f1.write(coordinate+','+label+'\n')
