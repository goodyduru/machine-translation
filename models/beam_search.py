import numpy as np
def beam_search(predict, k=1, maxsample=400, oov='oov', empty='empty', eos='eos'):
    dead_k = 0 
    dead_samples = []
    dead_scores = []

    live_k = 1
    live_samples = [[empty]]
    live_scores = [0]

    while live_k and dead_k < k:
        probabilities = predict(live_samples, empty)
        cand_scores = np.array(live_scores)[:, None] - np.log(probabilities)
        cand_scores[:, oov] = 1e20
        cand_flat = cand_scores.flatten()

        ranks_indices = cand_flat.argsort()[:(k - dead_k)]
        live_scores = cand_flat[ranks_indices]

        voc_size = probabilities.shape[1]
        live_samples = [live_samples[r//voc_size] + [r % voc_size] for r in ranks_indices]

        dead = [s[-1] == eos or len(s) >= maxsample for s in live_samples]

        dead_samples += [s for s, z in zip(live_samples, dead) if z]
        dead_scores += [s for s, z in zip(live_scores, dead) if z]
        dead_k = len(dead_samples)

        live_samples = [s for s, z in zip(live_samples, dead) if not z]
        live_scores = [s for s, z in zip(live_samples, dead) if not z]
        live_k = len(live_samples)

        return dead_samples + live_samples, dead_scores + dead_scores