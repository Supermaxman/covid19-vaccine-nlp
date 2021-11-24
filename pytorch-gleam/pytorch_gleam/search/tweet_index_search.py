
import json
from tqdm import tqdm
import argparse

from pyserini.search import SimpleSearcher


def batch(iterable, chunk_size=1):
	size = len(iterable)
	for ndx in range(0, size, chunk_size):
		yield iterable[ndx:min(ndx + chunk_size, size)]


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--index_path', required=True)
	parser.add_argument('-q', '--query_path', required=True)
	parser.add_argument('-r', '--output_path', required=True)
	parser.add_argument('-k', '--top_k', default=2000, type=int)
	parser.add_argument('-t', '--threads', default=8, type=int)
	parser.add_argument('-bk1', '--bm25_k1', default=0.82, type=float)
	parser.add_argument('-bb', '--bm25_b', default=0.68, type=float)

	args = parser.parse_args()

	with open(args.query_path) as f:
		questions = json.load(f)

	searcher = SimpleSearcher(args.index_path)
	searcher.set_bm25(args.bm25_k1, args.bm25_b)
	print(f'Running search...')

	queries = []
	for q_id, q in questions.items():
		q_txt = q['text']
		queries.append((q_id, q_txt))

	scores = {}
	batches = list(batch(queries, chunk_size=args.threads))
	for batch_pairs in tqdm(batches):
		batch_q_txt = [q_txt for q_p_id, q_txt in batch_pairs]
		batch_q_ids = [f'{q_p_id}' for q_p_id, q_txt in batch_pairs]

		q_hits = searcher.batch_search(
			queries=batch_q_txt,
			qids=batch_q_ids,
			k=args.top_k,
			threads=args.threads
		)
		for q_id, hits in q_hits.items():
			for rank, hit in enumerate(hits[:args.top_k], start=1):
				tweet_id = hit.docid
				if tweet_id not in scores:
					scores[tweet_id] = {}
				scores[tweet_id][q_id] = hit.score

	with open(args.output_path, 'w') as f:
		json.dump(scores, f)

	print('Done!')


if __name__ == '__main__':
	main()
