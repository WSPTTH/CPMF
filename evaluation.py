import numpy as np
from sklearn.metrics import roc_auc_score


def _results(reals, scores):
    """get the prediction and real label of a user"""
    res = []
    for eid in reals:
        res.append((reals[eid], scores[eid]))
    res.sort(key=lambda x: x[1], reverse=True)
    return res


class PrecisionK(object):

    def __init__(self, n=1):
        self._n = n

    def __call__(self, score):
        consider = score[:self._n]
        return sum((r for r, _ in consider)) / (self._n + 1e-20)


class NDCGK(object):

    def __init__(self, n=1):
        self._n = n

    def __call__(self, score):
        total_pos_num = sum((r for r, _ in score))
        pos_num = self._n if total_pos_num > self._n else total_pos_num

        idcg = sum((1 / np.log2(2 + ix) for ix in range(pos_num)))

        consider = score[:self._n]
        dcg = sum((1 / np.log2(ix + 2) for ix, (rx, _) in enumerate(consider) if rx > 0))
        return dcg / (idcg + 1e-20)


class AveragePrecision(object):

    def __call__(self, scores):
        total_pos_num = sum((r for r, _ in scores))
        count, ap = 0, 0
        for ix, (rx, _) in enumerate(scores, 1):
            count += rx
            ap += count / ix
            if ix > total_pos_num:
                break
        return ap / (total_pos_num + 1e-20)


class ReciprocalRank(object):

    def __call__(self, scores):
        total_pos_num = sum((r for r, _ in scores))
        rr = 0
        for ix, (rx, _) in enumerate(scores, 1):
            rr += 1 / ix
            if ix > total_pos_num:
                break
        return rr / (total_pos_num + 1e-20)


class AUC(object):

    def __call__(self, scores):
        real, score = list(zip(*scores))
        real = list(real)
        score = list(score)
        real.extend([0, 1])  # to avoid one label
        score.extend([0, 1])  # to avoid one label
        real = np.array(real, dtype=np.int)
        score = np.array(score)
        return roc_auc_score(real, score)


_Evaluator = {
    'P@1': PrecisionK(1),
    'P@3': PrecisionK(3),
    'P@5': PrecisionK(5),
    'P@10': PrecisionK(10),
    'NDCG@1': NDCGK(1),
    'NDCG@3': NDCGK(3),
    'NDCG@5': NDCGK(5),
    'NDCG@10': NDCGK(10),
    'MAP': AveragePrecision(),
    'MRR': ReciprocalRank(),
    'AUC': AUC()
}

_EVAL_KEYS = [
    'P@1', 'P@3', 'P@5', 'NDCG@1', 'NDCG@3', 'NDCG@5', 'MAP', 'MRR'
]


def evaluate_single_user(reals, scores):
    """evaluate the result of a single user"""
    scores = _results(reals, scores)
    res = {kx: _Evaluator[kx](scores) for kx in _EVAL_KEYS}
    return res


def evaluate(reals, scores):
    res = {}
    for uid in reals:
        u_res = evaluate_single_user(reals[uid], scores[uid])
        for kx in u_res:
            res[kx] = res.get(kx, 0.0) + u_res[kx]
    for kx in res:
        res[kx] /= (len(reals) + 1e-20)
    return res


def evaluate_auc(reals, scores):
    """the input is part of the data"""
    auc = 0.0
    auc_evaluator = AUC()
    for uid in reals:
        user_score = _results(reals[uid], scores[uid])
        auc += auc_evaluator(user_score)
    return {'AUC': auc / (len(reals) + 1e-20)}
