#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import re
import sys
#from gensim import corpora
import nltk
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize

punctuation = ['.', ',', ':', ';', '!', '?', '"', '\'', '(', ')','[', ']', '{', '}']
interrogative_pronouns = ['what time', 'what', 'when', 'where', \
							'which', 'who', 'whose', 'why', \
							'how many', 'how much', 'how']
num2word = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
            6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 19: 'Nineteen'}
num2word2 = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']

#need to be improved, like add 'would'
stopwords = [word.encode('utf-8') for word in stopwords.words("english")]
answer2index = {'A':1, 'B':2, 'C':3, 'D':4}

allvocabulary = []

def tokenizer(text):
	#'2.999.' -> '2' and '999'
	words = regexp_tokenize(text, pattern = r'\w+')
	return words

	#'2.999.' -> '2.999'
	#words = nltk.word_tokenize(text)
	#return [w for w in words if w not in punctuation]

#'number' '0'-'999' to words
def number2words(num):
	if num == 0:
		return 'Zero'
	elif num < 20:
		return num2word[num]
	elif num < 100:
		ray = divmod(num,10)
		if ray[1] == 0:
			return num2word2[ray[0]-2]
		else:
			return num2word2[ray[0]-2] + " " + num2word[ray[1]]
	elif num <1000:
		ray = divmod(num,100)
		if ray[1] == 0:
			mid = " hundred"
		else:
			mid =" hundred and "
		return num2word[ray[0]] + mid + number2words(ray[1])
	else:
		return str(num)

#use after replace entity as 'Owen' -> 'ow'
#maybe use to create vocabulary but 'cement' -> 'cem'
def stemming(word):
	#three ways
	#maybe better than others, but'owed' -> 'ow','cement' -> 'cem' and 'came' -> 'cam'
	lancaster_stemmer = LancasterStemmer()
	#'provision' -> 'provis', but'provide' -> 'provid'
	porter_stemmer = PorterStemmer()
	snowball_stemmer = SnowballStemmer("english")

	return lancaster_stemmer.stem(word) 

#maybe use to create vocabulary as 'cars' -> 'car' but 'has' -> 'ha'
def lemmatization(word):
	lemmatizer = WordNetLemmatizer()
	#lemmatizer.lemmatize(‘is’, pos=’v’) return u’be’
	return lemmatizer.lemmatize(word)

def contents_split(rawfile):
	with open(rawfile) as f:
		contents = f.read().split('\r\n')
		contents.remove('')
		
	cfiles_dir = './%s/%s/contents'	% (source, dataset)
	if not os.path.exists(cfiles_dir):
		os.mkdir(cfiles_dir)

	i = 0
	for content in contents:
		cfile = './%s/%s/contents/%d.tsv' % (source, dataset, i)
		#os.mknod(filename);???
		i += 1
		with open(cfile, "w") as f:
			f.write(content)

def answers_split(answersfile):
	with open(answersfile) as f:
		answers = f.read().split('\r\n')
		answers.remove('')

	afiles_dir = './%s/%s/answers'	% (source, dataset)
	if not os.path.exists(afiles_dir):
		os.mkdir(afiles_dir)

	i = 0
	for answer in answers:
		afile = './%s/%s/answers/%d.tsv' % (source, dataset, i)
		with open(afile, "w") as f:
			f.write(answer)
		i += 1

def extract_entity_names(tree):
    entity_names = []

    if hasattr(tree, 'label') and tree.label:
        if tree.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                entity_names.extend(extract_entity_names(child))

    return entity_names

def get_entities(story):
	entities = {}

	'''wrong code, before nltk.pos_tag(), 
		story need to be divide into sentences with',' and '.' using nltk.sent_tokenize(),
		then tokenize each sentence to tokens with ',' and '.' using nltk.word_tokenize.
	storytokens = tokenizer(story) #remove '\'', ',' and '.'
	pos_words = nltk.pos_tag(storytokens)
	'''

	sentences = nltk.sent_tokenize(story)
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	#label 'Boy' and 'Scout' as 'NNP' respectively 
	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
	#label 'Boy Scout' as 'NE'(entity)
	chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

	#
	entity_in_sentences = []
	for tree in chunked_sentences:
		#extract_entity_names(tree) find entities in each chunked_sentence
		entity_in_sentences.extend(extract_entity_names(tree))
	
	#delete repeat entities in all chunked_sentences
	entities_unique = set(entity_in_sentences)
	#create entities(dict object)
	i = 0
	for entity in entities_unique:
		entities[entity] = i
		i += 1

	return entities

def preprocess_story(story):
	#just '\\newline' will match ''
	story = re.sub(r'\\newline', '', story)
	story = re.sub(r'\\tab', '', story)
	#expand contractions, like don't to do not
	story = re.sub(r'n\'t', ' not', story)
	#print story

	#'number' '0'-'999' to words
	digit_re = re.compile(r'\d+')
	digits = digit_re.findall(story)
	for digit in digits:
		story = re.sub(digit, number2words(int(digit)), story)

	#print story

	#create vocabulary(dict object)
	i = 0
	vocabulary = {}
	storytokens = tokenizer(story.lower())
	storytokens = set(storytokens)
	for word in storytokens:
		#'cars' -> 'car' to keep just 'car' in vocabulary, but 'has' -> 'ha'
		#word = lemmatization(word)
		#'looked' -> 'look', but 'cement' -> 'cem'
		#word = stemming(word)

		#just ensure map vocabulary to consecutive numbers
		if not vocabulary.has_key(word):
			#not sure stopwords should be in vocabulary
			#if need, also can do this in tokenizer()
			#if word not in stopwords:
				i += 1
				vocabulary[word] = i
				allvocabulary.append(word)

	'''another way to create vocabulary with corpora in gensim
	words = [word for word in storytokens if word not in stopwords]
	corpora.Dictionary([words]).save(vfile)
	vocabulary = corpora.Dictionary.load(vfile)
	vocabulary = vocabulary.token2id.keys()
	'''

	entities = get_entities(story)
				
	#replace entity with @entity, 
	#need to be improved, entity maybe in a word 
	#like 'Dad' in 'Daddy' (may be fixed in token) and 
	#'Zebra' in 'Zebras' (may be fixed in lemma)
	for entity in entities.keys():	
		story = re.sub(entity, '@entity%d' % entities[entity], story)
	story_entity = story
	#print story_entity

	return story_entity, vocabulary, entities


def preprocess_question(question, entities):
	#remove 'one:', 'multiple:' and '?'
	question = question.lstrip('one: ').lstrip('multiple: ').rstrip('?')
	#expand contractions, like don't to do not
	question = re.sub(r'n\'t', ' not', question)
	#'number' '0'-'999' to words
	digit_re = re.compile(r'\d+')
	digits = digit_re.findall(question)
	for digit in digits:
		question = re.sub(digit, number2words(int(digit)), question)

	#replace entities obtained from story before lower()
	for entity in entities.keys():
		question = re.sub(entity, '@entity%d' % entities[entity], question)
	
	'''replace interrogative pronouns with '@placeholder', but can not handle two words like 'how much'
	questokens = tokenizer(question.lower())
	for token in questokens:
		if token in interrogative_pronouns:
			question = re.sub(token, '@placeholder', question)
	'''
	#replace interrogative pronouns with '@placeholder'
	for word in interrogative_pronouns:
		if question.lower().find(word)+1:
			question = re.sub(word, '@placeholder', question.lower())
			break

	'''remove stopwords, using token because ‘at’ in ‘cat’ 
		and notice questokenscopy
	questokens = tokenizer(question)
	for token in questokenscopy:
		if token in stopwords:
			questokens.remove(token)
	question = ' '.join(questokens).lower()
	'''

	return question

def preprocess_answer(answer, entities):
	#expand contractions, like don't to do not
	answer = re.sub(r'n\'t', ' not', answer)
	#'number' '0'-'999' to words
	digit_re = re.compile(r'\d+')
	digits = digit_re.findall(answer)
	for digit in digits:
		answer = re.sub(digit, number2words(int(digit)), answer)

	#replace entities obtained from story before lower()
	for entity in entities.keys():
		answer = re.sub(entity, '@entity%d' % entities[entity], answer)

	return answer.lower()

def get_answerindex(afile):
	index = []
	with open(afile) as f:
		answers = f.read().split('\t')

	for answer in answers:
		index.append(answer2index[answer])
		
	return index[0], index[1], index[2], index[3]

def get_s_q_a_e():
	qfiles_dir = './%s/%s/questions' % (source, dataset)
	if not os.path.exists(qfiles_dir):
		os.mkdir(qfiles_dir)

	vfiles_dir = './%s/%s/vocabularies'	% (source, dataset)
	if not os.path.exists(vfiles_dir):
		os.mkdir(vfiles_dir)

	i = 0
	cfile = './%s/%s/contents/%d.tsv' % (source, dataset, i)

	while (os.path.exists(cfile)):
		afile = './%s/%s/answers/%d.tsv' % (source, dataset, i)
		vfile = './%s/%s/vocabularies/%d.tsv' % (source, dataset, i)

		with open(cfile, "r") as f:
			content = f.read()

		content = content.split('\t')
		info = content[1]
		story_entity, vocabulary, entities = preprocess_story(content[2])

		with open(vfile, "w") as f:
			f.write('\n'.join(vocabulary))	

		i1, i2, i3, i4 = get_answerindex(afile)

		q1 = preprocess_question(content[3], entities)
		answer1 = preprocess_answer(content[3 + i1], entities)

		q2 = preprocess_question(content[8], entities)
		answer2 = preprocess_answer(content[8 + i2], entities)

		q3 = preprocess_question(content[13], entities)
		answer3 = preprocess_answer(content[13 + i3], entities)

		q4 = preprocess_question(content[18], entities)
		answer4 = preprocess_answer(content[18 + i4], entities)

		e_list = []
		for entity in entities.keys():
			e_list.append('@entity%d' % entities[entity] + ':' + entity)
		entity_list  = '\n'.join(e_list)

		info_story = info + '\n\n' + story_entity + '\n\n' 
		question1 = q1 + '\n\n' + answer1 + '\n\n' 
		question2 = q2 + '\n\n' + answer2 + '\n\n' 
		question3 = q3 + '\n\n' + answer3 + '\n\n' 
		question4 = q4 + '\n\n' + answer4 + '\n\n' 

		#four questions in one file
		qfile = './%s/%s/questions/%d.tsv' % (source, dataset, i)
		question = info_story + question1 + question2 + question3 + question4 + entity_list
		with open(qfile, "w") as f:
			f.write(question)
		
		#each questions in one file
		q = []			
		question1 = info_story + question1 + entity_list
		question2 = info_story + question2 + entity_list
		question3 = info_story + question3 + entity_list
		question4 = info_story + question4 + entity_list
		q.append(question1) 
		q.append(question2)
		q.append(question3) 
		q.append(question4) 

		for j in range(0, 4):
			qfile = './%s/%s/questions/%d_q%s.tsv' % (source, dataset, i, j+1)
			with open(qfile, "w") as f:
				f.write(q[j])
		
		i += 1
		cfile = './%s/%s/contents/%d.tsv' % (source, dataset, i)

	#ensure no repeat word in dictionary
	dictionary = set(allvocabulary)
	dfile = './%s/%s/dictionary.txt' % (source, dataset)
	with open(dfile, "w") as f:
		f.write('\n'.join(dictionary))

#use after datasets have all been preprocessed
def create_dictionary(source):
	dictionary = []
	datasets = ['train', 'dev', 'test']
	for dataset in datasets:
		dfile = './%s/%s/dictionary.txt' % (source, dataset)
		with open(dfile, "r") as f:
			dictionary.extend(f.read().split('\n'))
	dictionary = set(dictionary)
	filename = './%s/dictionary.txt' % (source)
	with open(filename, "w") as f:
		f.write('\n'.join(dictionary))

def main(source, dataset):
	rawfile = './MCTest/%s.%s.tsv' % (source, dataset)
	answersfile = './MCTest/%s.%s.ans' % (source, dataset)

	source_dir = './%s'	% (source)
	if not os.path.exists(source_dir):
		os.mkdir(source_dir)
	dataset_dir = './%s/%s'	% (source, dataset)
	if not os.path.exists(dataset_dir):
		os.mkdir(dataset_dir)

	contents_split(rawfile)
	answers_split(answersfile)
	get_s_q_a_e()

if __name__ == '__main__':
	if len(sys.argv) == 3:
	    source = sys.argv[1]
	    dataset = sys.argv[2]
	    main(source, dataset)
	elif len(sys.argv) == 2:
	    source = sys.argv[1]
	    create_dictionary(source)
	else:
	    print(" [*] usage: python preprocess_data.py source(mc160|mc500) dataset(train|dev|test)")
      
