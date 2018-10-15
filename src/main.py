import requests
import json
from argparse import ArgumentParser
from progressbar import ProgressBar

import numpy as np


def main():
    parser = ArgumentParser()

    parser.add_argument("--train_file", type=str, default="")
    parser.add_argument("--dev_file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")

    args = parser.parse_args()

    '''
    lines = []

    with open(args.file) as f:
        for line in f:
            lines.append(line)

    data = " ".join(lines)

    with open(args.output_file, 'w') as of:
        of.write(data)
        of.write('\n')
    '''

    # data = json.load(open(args.file, 'r'))

    ''' 
    ##oxford dictionary API
    # API key
    app_id = '984b033a'
    app_key = "[my-key]""

    language = 'en'
    # word_id = 'test'

    # url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/' + language + '/' + word_id.lower()
    base_url = 'https://od-api.oxforddictionaries.com:443/api/v1/entries/' + language + '/'
    '''

    ''''''
    ##wordnik
    api_key = "[my-key]"
    # word_id = 'test'
    # prop_str = 'limit=200&includeRelated=false&sourceDictionaries=all&useCanonical=false&includeTags=false'
    base_url = 'https://api.wordnik.com/v4/word.json/'
    # + word_id.lower() + '/definitions'
    request_type = '/definitions'
    # + '?' + prop_str + '&api_key=' + api_key

    '''
    # get words
    '''
    word_set = set()

    print("load training data...\n")
    with open(args.train_file, 'r') as f:
        json_list = [json.loads(line) for line in f]

        pbar = ProgressBar()
        for json_data in pbar(json_list):
            word_set |= set(json_data['document'])
            word_set |= set(json_data['question'])

    print("load dev data...\n")
    with open(args.dev_file, 'r') as f:
        json_list = [json.loads(line) for line in f]

        pbar = ProgressBar()
        for json_data in pbar(json_list):
            word_set |= set(json_data['document'])
            word_set |= set(json_data['question'])

    print('word size={}\n'.format(len(word_set)))

    # r = requests.get(url, headers={'app_id': app_id, 'app_key': app_key})

    # print("code {}\n".format(r.status_code))
    # print("text \n" + r.text)
    # print("json \n" + json.dumps(r.json()))

    '''
    r = requests.get(base_url,
                     headers={'limit': '200', 'api_key': api_key, 'includeRelated': 'false',
                              'sourceDictionaries': 'all',
                              'useCanonical': 'false', 'includeTags': 'false'})

    print("code {}\n".format(r.status_code))
    print("text \n" + r.text)
    print("json \n" + json.dumps(r.json()))
    '''

    # --train_file ../../09_Reinforced_Mnemonic_Reader_For_Machine_Reading/00_mreader/src/SQuAD-train-v1.1-processed-spacy.txt --dev_file ../../09_Reinforced_Mnemonic_Reader_For_Machine_Reading/00_mreader/src/SQuAD-dev-v1.1-processed-spacy.txt --output-file ../data/wordnik_output.json
    print("write word definitions...\n")
    with open(args.output_file, 'w') as of:

        pbar = ProgressBar()
        for word in pbar(word_set):
            url = base_url + word.lower() + request_type
            r = requests.get(url,
                             headers={'limit': '200', 'api_key': api_key, 'includeRelated': 'false',
                                      'sourceDictionaries': 'all',
                                      # 'useCanonical': 'false',
                                      'useCanonical': 'true',
                                      'includeTags': 'false'})

            if r.status_code == 200 and r.text.strip() != '':
                of.write(json.dumps(r.json()))
                of.write('\n')


# print(data)

def reform():
    # --input-file ../data/wordnik_output.json --output-file ../data/wordnik_output_reform.json
    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")

    args = parser.parse_args()

    with open(args.input_file, 'r') as in_f, \
            open(args.output_file, 'w') as o_f:
        for line in in_f:
            if line.strip() == '[]':
                continue
            else:
                o_f.write(line)


def convert_to_json():
    # --input-file ../data/wordnik_output_c_reform.json --output-file ../data/wordnik_output_c_reform_new.json
    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--output-file", type=str, default="")

    args = parser.parse_args()

    with open(args.input_file, 'r') as in_f, \
            open(args.output_file, 'w') as o_f:
        dict = {}
        for line in in_f:
            try:
                data = json.loads(line)
                word = data[0]['word']
                dict[word] = data
            except Exception as e:
                print("error happened:{}\n".format(word))
                # print("\n")

        json.dump(dict, o_f)


def calc_word_coverage():
    # --input-file ../data/wordnik_output_c_reform_new.json --embedding-file ../data/glove.6B.100d.txt
    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--embedding-file", type=str, default="")

    args = parser.parse_args()

    embedding_dict = {}
    embedding_set = set()
    with open(args.embedding_file, 'r') as f:
        pbar = ProgressBar()

        for line in pbar(f):
            vector = line.rstrip().split(' ')
            word = vector[0]

            embedding = vector[1:]

            assert (len(embedding) == 300)
            embedding_dict[word.lower()] = embedding
            embedding_set.add(word)

    # count = 0
    json_data = json.load(open(args.input_file))
    voc_set = set()

    for key in json_data:
        item = json_data[key]

        size = len(item)
        for i in range(size):
            if 'text' not in item[i]:
                continue
            voc_set |= set(item[i]['text'].strip().split())

    print(len(voc_set))
    print(len(embedding_set))

    print("ratio(i/e):{}\n".format(float(len(voc_set.intersection(embedding_set))) / len(embedding_set)))
    print("ratio(i/v):{}\n".format(float(len(voc_set.intersection(embedding_set))) / len(voc_set)))

    return


def calc_definition_word_coverage():
    # --input-file ../data/wordnik_output_c_reform_new.json --embedding-file ../data/glove.6B.100d.txt
    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--embedding-file", type=str, default="")
    parser.add_argument("--output-vocab-file", "-o", type=str, default="")

    args = parser.parse_args()

    embedding_dict = {}
    embedding_set = set()
    with open(args.embedding_file, 'r') as f:
        pbar = ProgressBar()

        for line in pbar(f):
            vector = line.rstrip().split(' ')
            word = vector[0]

            embedding = vector[1:]

            assert (len(embedding) == 300)
            embedding_dict[word.lower()] = embedding
            embedding_set.add(word)

    # count = 0
    json_data = json.load(open(args.input_file))
    voc_set = set()
    count = 0

    for key in json_data:
        item = json_data[key]

        size = len(item)

        has_oov = False
        for i in range(size):
            if 'text' not in item[i]:
                continue

            text = item[i]["text"]
            for j in text.strip().split():
                if j not in embedding_set:
                    has_oov = True
                    break

            if has_oov:
                break

        if has_oov is False:
            count += 1
            voc_set.add(key)

    print("ratio(defintion/word):{}\n".format(float(count) / len(json_data)))
    # voc_set |= set(item[i]['text'].strip().split())

    with open(args.output_vocab_file, "w") as f:
        for item in voc_set:
            f.write(item)
            f.write("\n")
    """
    print(len(voc_set))
    print(len(embedding_set))

    print("ratio(i/e):{}\n".format(float(len(voc_set.intersection(embedding_set))) / len(embedding_set)))
    print("ratio(i/v):{}\n".format(float(len(voc_set.intersection(embedding_set))) / len(voc_set)))
    """

    return


def definition_process_json():
    # --input-file ../data/wordnik_output_c_reform_new.json --input-vocab-file ../data/vocab_coverage.txt --embedding-file ../../mreader/data/embeddings/glove.840B.300d.txt -oj ../data/definitions.json -oe ../data/definitions_embedding
    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--input-vocab-file", type=str, default="")
    parser.add_argument("--embedding-file", type=str, default="")
    parser.add_argument("--output-json-file", "-oj", type=str, default="")
    parser.add_argument("--output-embedding-file", "-oe", type=str, default="")

    args = parser.parse_args()

    embedding_dict = {}
    embedding_set = set()
    with open(args.embedding_file, 'r') as f:
        pbar = ProgressBar()

        for line in pbar(f):
            vector = line.rstrip().split(' ')
            word = vector[0]

            embedding = vector[1:]

            assert (len(embedding) == 300)
            embedding_dict[word.lower()] = embedding
            embedding_set.add(word)

    # count = 0
    json_data = json.load(open(args.input_file))
    vocab_set = set()
    vocab_dict = {}

    # collect definition vocab
    with open(args.input_vocab_file) as f:
        for line in f:
            if line.strip() != "":
                json_item = json_data[line.strip()]
                vocab_dict[line.strip()] = json_item
                for item in json_item:
                    vocab_set |= set(item["text"].strip().split())

    print("length of vocab_set:{}".format(len(vocab_set)))
    print("length of vocab_dict:{}".format(len(vocab_dict)))

    # create embeddings
    embedding_matrix = np.zeros((len(vocab_set) + 1, 300), dtype=np.float32)

    vocab_index = {}
    for i in range(1, len(vocab_set) + 1):
        word = list(vocab_set)[i - 1]
        embedding_matrix[i, :] = embedding_dict[word.lower()]
        vocab_index[word] = i

    vocab_seq_dict = {}

    for key in vocab_dict:
        json_item = vocab_dict[key]
        vocab_seq_list = []
        for item in json_item:
            definition = item["text"].strip().split()
            definition_ids = [str(vocab_index[word]) for word in definition]
            vocab_seq_list.append({"definition_ids": ' '.join(definition_ids)})
        vocab_seq_dict[key] = vocab_seq_list

    embedding_matrix.dump(args.output_embedding_file)

    with open(args.output_json_file, "w") as of:
        json.dump(vocab_seq_dict, of)

    return


def definition_process_all_json():
    # --input-file ../data/wordnik_output_c_reform_new.json --embedding-file ../../mreader/data/embeddings/glove.840B.300d.txt -oj ../data/definitions.json -oe ../data/definitions_embedding
    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--embedding-file", type=str, default="")
    parser.add_argument("--output-json-file", "-oj", type=str, default="")
    parser.add_argument("--output-embedding-file", "-oe", type=str, default="")

    args = parser.parse_args()

    embedding_dict = {}
    embedding_set = set()
    with open(args.embedding_file, 'r') as f:
        pbar = ProgressBar()

        for line in pbar(f):
            vector = line.rstrip().split(' ')
            word = vector[0]

            embedding = vector[1:]

            assert (len(embedding) == 300)
            embedding_dict[word.lower()] = embedding
            embedding_set.add(word)

    # count = 0
    json_data = json.load(open(args.input_file))
    vocab_set = set()
    vocab_dict = {}

    # count = 0
    # vocab_index ={}

    print("collect vocab...\n")
    pbar = ProgressBar()
    for key in pbar(json_data):

        json_item = json_data[key]

        for item in json_item:
            if "text" not in item:
                continue

            words = item["text"].strip().split()
            for word in words:
                if word in vocab_set:
                    continue

                if word in embedding_dict or word.lower() in embedding_dict:
                    vocab_set.add(word)
                    # vocab_dict[word] = count
                    # count += 1

    """
    # collect definition vocab
    with open(args.input_vocab_file) as f:
        for line in f:
            if line.strip() != "":
                json_item = json_data[line.strip()]
                vocab_dict[line.strip()] = json_item
                for item in json_item:
                    vocab_set |= set(item["text"].strip().split())

    """
    print("length of vocab_set:{}".format(len(vocab_set)))
    print("length of vocab_dict:{}".format(len(vocab_dict)))

    # create embeddings
    embedding_matrix = np.zeros((len(vocab_set) + 1, 300), dtype=np.float32)
    embedding_matrix.dump(args.output_embedding_file)

    vocab_index = {}
    for i in range(1, len(vocab_set) + 1):
        word = list(vocab_set)[i - 1]
        embedding_matrix[i, :] = embedding_dict[word.lower()]
        vocab_index[word] = i

    vocab_seq_dict = {}

    """
    for key in vocab_dict:
        json_item = vocab_dict[key]
        vocab_seq_list = []
        for item in json_item:
            definition = item["text"].strip().split()
            definition_ids = [str(vocab_index[word]) for word in definition]
            vocab_seq_list.append({"definition_ids": ' '.join(definition_ids)})
        vocab_seq_dict[key] = vocab_seq_list
    """

    max_defi_length = 0

    pbar = ProgressBar()
    print("create seq list...\n")
    for key in pbar(json_data):

        json_item = json_data[key]
        vocab_seq_list = []

        for item in json_item:
            if "text" not in item:
                continue

            words = item["text"].strip().split()

            """
            for word in words:
                if word in vocab_set:
                    vocab_seq_list.append(vocab_index[word])
                else:
                    vocab_seq_list.append(0)

            """

            definition_ids = [str(vocab_index[word]) if word in vocab_set else '0' for word in words]
            if len(definition_ids) > max_defi_length:
                max_defi_length = len(definition_ids)

            vocab_seq_list.append({'definition_ids': ' '.join(definition_ids)})

        if len(vocab_seq_list) == 0:
            continue

        vocab_seq_dict[key] = vocab_seq_list

    with open(args.output_json_file, "w") as of:
        json.dump(vocab_seq_dict, of)

    print("max definition length={}\n".format(max_defi_length)) # max definition length = 127

    return


if __name__ == '__main__':
    # main()
    # reform()
    # convert_to_json()
    # calc_word_coverage()
    # calc_definition_word_coverage()
    # definition_process_json()
    definition_process_all_json()
