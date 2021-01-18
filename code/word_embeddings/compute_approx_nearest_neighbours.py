# TODO: Implement

import annoy

"""
a = annoy.AnnoyIndex(f=last_embedding_weights.shape[1], metric="euclidean")

for i in tqdm(range(len(last_embedding_weights))):
    a.add_item(i, last_embedding_weights_normalized[i])

a.build(n_trees=100, n_jobs=-1)

a.save("test_annoy_index.ann")

a = annoy.AnnoyIndex(f=last_embedding_weights.shape[1], metric="euclidean")
a.load("test_annoy_index.ann")

words[a.get_nns_by_item(655, 10)]
"""
