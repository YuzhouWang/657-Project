import os

trainfile = './MCTest/mc500.train.tsv'

def readtsv(filename):
	with open(filename) as f:
		content = f.read()
		return content.split('\r\n')

def writetsvs(stories):
	i = 0
	files_dir = './stories'
	if not os.path.exists(files_dir):
		os.mkdir(files_dir)
	for story in stories:
		filename = './stories/%d.tsv' % i
		#os.mknod(filename);???
		i += 1
		with open(filename, "w") as f:
			f.write(story)

def main():
	stories = readtsv(trainfile)
	stories.remove('');
	writetsvs(stories)

if __name__ == '__main__':
	main()