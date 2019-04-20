import json
import os
import numpy as np
from tqdm import tqdm
from evaluation import evaluate, evaluate_auc


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ssig(x):
    sx = sigmoid(x)
    return sx * (sx - 1)


class CSPF(object):
    """
    puij ~ laplace(sigmoid(x_uij), b)
    regularization: ||\cdot||_F^2
    """

    def __init__(self, dataset, save_path, k=50, rho_g=5, rho_l=3,
                 iter_num=1e7, save_step=1e4, lr=0.05, gamma=0.05, batch_size=64):
        """
        :param dataset: the input dataset, from the dataset.Dataset
        :param iter_num: the iteration number
        :param save_path: the path to save model
        :param save_step: the step to save model
        :param lr: the learning rate
        :param gamma: the regularization parameter
        :param k: the dimension of all the vectors
        :param rho_g: the loss ratio for group
        :param rho_l: the loss ratio for location
        :param batch_size: the batch size
        """
        self._dataset = dataset
        self._iter_num = int(iter_num)
        self._save_step = save_step
        self._save_path = save_path

        self._dim = k
        self._rho_g = rho_g
        self._rho_l = rho_l
        self._lr = lr
        self._gamma = gamma
        self._batch_size = batch_size

        self._is_train = False

        # initialize the parameters
        self._U = np.random.rand(len(self._dataset.users) + 1, self._dim)
        self._E = np.random.rand(len(self._dataset.events) + 1, self._dim)
        self._L = np.random.rand(len(self._dataset.locations) + 1, self._dim)
        self._G = np.random.rand(len(self._dataset.groups) + 1, self._dim)

        self._U_mapping = {ux: ix for ix, ux in enumerate(self._dataset.users, 1)}
        self._E_mapping = {ex: ix for ix, ex in enumerate(self._dataset.events, 1)}
        self._L_mapping = {lx: ix for ix, lx in enumerate(self._dataset.locations, 1)}
        self._G_mapping = {gx: ix for ix, gx in enumerate(self._dataset.groups, 1)}

    def save(self):
        all_values = {
            'user': self._U, 'event': self._E, 'location': self._L, 'group': self._G
        }

        l_mapping_new = {str(kx): val for kx, val in self._L_mapping.items()}

        all_mapping = {
            'user': self._U_mapping, 'event': self._E_mapping,
            'location': l_mapping_new, 'group': self._G_mapping
        }
        base_dir = os.path.split(self._save_path)[0]
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        np.savez_compressed(self._save_path, **all_values)
        with open(self._save_path + '.mapping', 'w', encoding='utf-8') as fp:
            json.dump(all_mapping, fp, ensure_ascii=False)
        return self

    def load(self):
        with np.load(self._save_path) as data:
            self._U = data['user']
            self._E = data['event']
            self._G = data['group']
            self._L = data['location']

        with open(self._save_path + '.mapping', 'r', encoding='utf-8') as fp:
            all_mapping = json.load(fp)
            self._U_mapping = all_mapping['user']
            self._E_mapping = all_mapping['event']
            self._L_mapping = {eval(kx): val for kx, val in all_mapping['location'].items()}
            self._G_mapping = all_mapping['group']

        return self

    def _mapping_data(self, batch_data):
        """mapping data with dicts"""
        u_index = np.array([self._U_mapping.get(ix, 0) for ix in batch_data.u])

        ep_index = np.array([self._E_mapping.get(ix, 0) for ix in batch_data.ep])
        en_index = np.array([self._E_mapping.get(ix, 0) for ix in batch_data.en])

        gp_index = np.array([self._G_mapping.get(ix, 0) for ix in batch_data.gp])
        gn_index = np.array([self._G_mapping.get(ix, 0) for ix in batch_data.gn])

        lp_index = np.array([self._L_mapping.get(ix, 0) for ix in batch_data.lp])
        ln_index = np.array([self._L_mapping.get(ix, 0) for ix in batch_data.ln])

        ugr = np.array(batch_data.ugr).reshape(-1, 1)
        ulr = np.array(batch_data.ulr).reshape(-1, 1)
        return u_index, ep_index, en_index, gp_index, gn_index, lp_index, ln_index, ugr, ulr

    def _train_step(self, batch_data):
        # get the batch data
        u_index, ep_index, en_index, gp_index, gn_index, lp_index, ln_index, ugr, ulr = self._mapping_data(batch_data)

        # get latent vector
        u = self._U[u_index]
        ep, en = self._E[ep_index], self._E[en_index]
        gp, gn = self._G[gp_index], self._G[gn_index]
        lp, ln = self._L[lp_index], self._L[ln_index]

        # compute the gradient to update parameter
        # initialize the gradient
        du = np.zeros_like(u)
        dep, den = np.zeros_like(ep), np.zeros_like(en)
        dgp, dgn = np.zeros_like(gp), np.zeros_like(gn)
        dlp, dln = np.zeros_like(lp), np.zeros_like(ln)

        # from user-event pairs
        ue_value = ssig(np.sum(u * (ep - en), axis=1, keepdims=True))
        du += ue_value * (ep - en)
        dep += ue_value * u
        den += - ue_value * u

        # from user-group pairs
        ug_val = ssig(np.sum(u * (gp - gn) * ugr, axis=1, keepdims=True)) * ugr
        du += self._rho_g * ug_val * (gp - gn)
        dgp += self._rho_g * ug_val * u
        dgn += - self._rho_g * ug_val * u

        # from user-location pairs
        ul_val = ssig(np.sum(u * (lp - ln) * ulr, axis=1, keepdims=True)) * ulr
        du += self._rho_l * ul_val * (lp - ln)
        dlp += self._rho_l * u
        dln += - self._rho_l * u

        # regularization
        du += self._gamma * u
        dep += self._gamma * ep
        den += self._gamma * en
        dgp += self._gamma * gp
        dgn += self._gamma * gn
        dlp += self._gamma * lp
        dln += self._gamma * ln

        # update parameter
        self._U[u_index] -= self._lr * du
        self._E[ep_index] -= self._lr * dep
        self._E[en_index] -= self._lr * den
        self._G[gp_index] -= self._lr * dgp
        self._G[gn_index] -= self._lr * dgn
        self._L[lp_index] -= self._lr * dlp
        self._L[ln_index] -= self._lr * dln

    def _score(self, u, g, l):
        """given a batch data, compute the score, u, g, l are all ids"""
        u_index = np.array([self._U_mapping.get(ix, 0) for ix in u])
        g_index = np.array([self._G_mapping.get(ix, 0) for ix in g])
        l_index = np.array([self._L_mapping.get(ix, 0) for ix in l])

        uvec = self._U[u_index]
        gvec = self._G[g_index]
        lvec = self._L[l_index]

        gs = np.sum(uvec * gvec, axis=1)
        ls = np.sum(uvec * lvec, axis=1)
        score = (self._rho_g * gs + self._rho_l * ls) / (self._rho_g + self._rho_l + 1e-20)
        return score

    def  _base_eval(self, sample_gen, label=1, name='', logx=''):
        scores, reals = {}, {}
        for u, e, g, l in tqdm(sample_gen, ascii=True, leave=False,
                               desc='[eval] computing score for `{}`: {}'.format(name, logx)):
            batch_score = self._score(u, g, l)
            for ux, ex, sx in zip(u, e, batch_score):
                if ux not in scores:
                    scores[ux] = {}
                scores[ux][ex] = sx
                if ux not in reals:
                    reals[ux] = {}
                reals[ux][ex] = label
        return scores, reals

    @staticmethod
    def _data_combine(pos, neg):
        res = {}
        for kx in pos:
            res[kx] = pos[kx].copy()
            if kx in neg:
                res[kx].update(neg[kx])
        for kx in (neg.keys() - pos.keys()):
            res[kx] = neg[kx].copy()
        return res

    def eval(self, evals):
        """evaluate the test dataset, user by user"""
        assert isinstance(evals, (dict, list, tuple))
        if len(evals) == 0:
            return
        if isinstance(evals[0], list):
            eval_iter = evals
        else:
            eval_iter = enumerate(evals)

        res = []
        for name, dev in eval_iter:
            if not dev.is_test:
                continue  # is not the test dataset
            # positive dataset
            sp, rp = self._base_eval(dev.test_samples(batch=self._batch_size, mode='pos'), 1, name, 'positive')
            # negative dataset
            sn, rn = self._base_eval(dev.test_samples(batch=self._batch_size, mode='neg'), 0, name, 'negative')
            # auc negative dataset
            san, ran = self._base_eval(dev.test_samples(batch=self._batch_size, mode='auc'), 0, name, 'auc negative')

            scores = self._data_combine(sp, sn)
            reals = self._data_combine(rp, rn)
            rank_res = evaluate(reals, scores)

            score_auc = self._data_combine(sp, san)
            real_auc = self._data_combine(rp, ran)
            rank_res.update(evaluate_auc(real_auc, score_auc))

            res.append((name, rank_res))

        return res

    def train(self, evals=None):
        if self._is_train:
            return

        if evals is not None:
            if isinstance(evals, dict):
                evals = list(evals.items())
            if not isinstance(evals, (list, tuple)):
                raise ValueError('The `evals` should be a list of the dataset class')

        train_samples = self._dataset.batch(self._batch_size)
        for ix in tqdm(range(self._iter_num), desc='training', ascii=True):
            self._train_step(next(train_samples))
            if (ix + 1) % self._save_step == 0:
                if evals is not None:
                    all_res = self.eval(evals)
                    for name, res in all_res:
                        log_base = '[Evaluate at {:6d}] Dataset `{}` | '.format(ix + 1, name)
                        str_res = ['{}: {:>1.5f}'.format(kx, vx) for kx, vx in sorted(res.items())]
                        log_values = ' | '.join(str_res)
                        tqdm.write(log_base + log_values)
                self.save()
        return self
