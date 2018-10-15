import tensorflow as tf
import tensorflow_hub as hub

from argparse import ArgumentParser
import json
import numpy as np

from progressbar import ProgressBar

"""
embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
embeddings = embed([
    "The quick brown fox jumps over the lazy dog.",
    "I am a sentence for which I would like to get its embedding"])

print session.run(embeddings)
"""

if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--embedding-file", type=str, default="")
    parser.add_argument("--output-json-file", "-oj", type=str, default="")
    parser.add_argument("--output-embedding-file", "-oe", type=str, default="")

    args = parser.parse_args()

    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

    seq_id = 0
    seq_embedding_id_dict = {}
    embeddings = {}

    json_data = json.load(open(args.input_file))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        """
        pbar = ProgressBar()
        for key in pbar(json_data):

            json_item = json_data[key]
            seq_ids = []

            for item in json_item:

                if "text" not in item:
                    continue

                embedding = embed([item["text"].strip()])
                embeddings[seq_id] = sess.run(embedding)
                seq_ids.append({"definition": seq_id})
                seq_id += 1

            seq_embedding_id_dict[key] = seq_ids
        """

        pbar = ProgressBar()
        embedding_to_run = []
        for key in pbar(json_data):

            json_item = json_data[key]
            seq_ids = []

            for item in json_item:

                if "text" not in item:
                    continue

                # embedding = embed([item["text"].strip()])
                embedding_to_run.append(item["text"].strip())
                # embeddings[seq_id] = sess.run(embedding)
                seq_ids.append({"definition": seq_id})
                seq_id += 1

            seq_embedding_id_dict[key] = seq_ids
        emb = embed(embedding_to_run)
        embeddings = sess.run(emb)

    embedding_matrix = np.zeros((seq_id, 512), dtype=np.float32)

    for i in range(seq_id):
        embedding_matrix[i, :] = embeddings[i][0]

    # with open(args.output_embedding_file, "w") as of:
    embedding_matrix.dump(args.output_embedding_file)

    with open(args.output_json_file, "w") as of:
        json.dump(seq_embedding_id_dict, of)
