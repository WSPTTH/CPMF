import json
import random
from collections import namedtuple
import numpy as np


def _sign(x):
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    else:
        return 0.0


class _Pairs(object):

    def __init__(self):
        self._pairs = []
        self._front_head = {}
        self._front_list = {}

    def _build_sample_list(self):
        """convert the dict into list for sampling"""
        for front in self._front_head:
            self._front_list[front] = list(self._front_head[front].keys())
        return self

    def fit(self, pairs):
        self._pairs = pairs.copy()
        for f, t in self._pairs:
            if f not in self._front_head:
                self._front_head[f] = {}
            self._front_head[f][t] = self._front_head[f].get(t, 0) + 1
        self._build_sample_list()
        return self

    def transition(self, fm, mt, neg_thre=2, weight=1):
        """
        transform the two-hop links
        :param fm: the _Pair instance of with the same front set to a middle set
        :param mt: the _Pair instance of with the same tail set from the same middle set of `fm`
        :param neg_thre: transition threshold
        :param weight: the weight of the transition
        """
        assert isinstance(fm, _Pairs) and isinstance(mt, _Pairs)
        for front in fm.front_dict:
            if front not in self._front_head:
                continue
            for mid, vfm in fm.front_dict[front].items():
                if mid not in mt.front_dict:
                    continue
                for tail, vmt in mt.front_dict[mid].items():
                    self._front_head[front][tail] = self._front_head[front].get(tail, 0) + weight * vfm * vmt
        for front in self._front_head:
            for tail in self._front_head[front]:
                if self._front_head[front][tail] < neg_thre:
                    self._front_head[front][tail] = -1
        self._build_sample_list()
        return self

    @property
    def pairs(self):
        return self._pairs

    @property
    def front_dict(self):
        return self._front_head

    @property
    def front_list(self):
        return self._front_list

    def __len__(self):
        return len(self._pairs)


_Sample = namedtuple('_Sample', ('u', 'ep', 'en', 'gp', 'gn', 'lp', 'ln', 'ugr', 'ulr'))


class Dataset(object):

    @staticmethod
    def _filter(pairs, front, tail):
        """filter the pairs by the given front set and tail set"""
        return [(t, l) for t, l in pairs if t in front and l in tail]

    @staticmethod
    def _front_tail_set(pairs):
        front_set, tail_set = set(), set()
        for fx, tx in pairs:
            front_set.add(fx)
            tail_set.add(tx)
        return front_set, tail_set

    def _component_set(self, ue, ug, eg, ul, el):
        user_ue, event_ue = self._front_tail_set(ue)
        user_ug, group_ug = self._front_tail_set(ug)
        event_eg, group_eg = self._front_tail_set(eg)
        user_ul, loc_ul = self._front_tail_set(ul)
        event_el, loc_el = self._front_tail_set(el)

        users = user_ue & user_ug & user_ul
        events = event_ue & event_eg & event_el
        groups = group_eg | group_ug
        locs = loc_ul | loc_el
        return users, events, groups, locs

    def __init__(self, trans_thre=2):
        self._trans_thre = trans_thre

    def build(self, ue, ug, eg, ul, el, neg_ue):
        ue_all = ue + neg_ue
        self._users, self._events, self._groups, self._locations = self._component_set(ue_all, ug, eg, ul, el)
        self._ue = _Pairs().fit(self._filter(ue, self._users, self._events))
        self._ug = _Pairs().fit(self._filter(ug, self._users, self._groups))
        self._eg = _Pairs().fit(self._filter(eg, self._events, self._groups))
        self._ul = _Pairs().fit(self._filter(ul, self._users, self._locations))
        self._el = _Pairs().fit(self._filter(el, self._events, self._locations))

        self._neg_ue = _Pairs().fit(self._filter(neg_ue, self._users, self._events))

        # add the transition information
        self._ug.transition(self._ue, self._eg, neg_thre=self._trans_thre)
        self._ul.transition(self._ue, self._el, neg_thre=self._trans_thre)

        self._is_test = True if neg_ue is not None else False
        self._users = list(self._users)
        self._locations = list(self._locations)
        self._events = list(self._events)
        self._groups = list(self._groups)

        self._sample_user_list = list(self._ue.front_list.keys())
        print('positive pairs: ', len(self._ue.pairs))
        print('negative pairs: ', len(self._neg_ue.pairs))
        return self

    def save(self, save_path):
        save_all = {
            'ue': self._ue.pairs,
            'ug': self._ug.pairs,
            'ul': self._ul.pairs,
            'eg': self._eg.pairs,
            'el': self._el.pairs,
            'neg_ue': self._neg_ue.pairs
        }
        with open(save_path, 'w', encoding='utf-8') as fp:
            json.dump(save_all, fp)
        return self

    def load(self, save_path):
        with open(save_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        ul = [(ux, tuple(locx)) for ux, locx in data['ul']]
        el = [(ex, tuple(locx)) for ex, locx in data['el']]
        self.build(data['ue'], data['ug'], data['eg'], ul, el, data['neg_ue'])
        return self

    @property
    def users(self):
        return self._users

    @property
    def events(self):
        return self._events

    @property
    def groups(self):
        return self._groups

    @property
    def locations(self):
        return self._locations

    @property
    def is_test(self):
        return self._is_test

    def _sample(self):
        """
        Sampling a sample pair from the dataset. A data sample contains a positive sample and negative sample.
        A sample contains:
            - a user `u`
            - a positive event `ei` and a negative event `ej`
            - the group `gi` of `ei`, and the group `gj` of `ej`
            - the location `li` of `ei`, and  the location `lj` of `ej`
            - the user-group relationship
            - the user-location relationship
        """
        # sample a user
        u = random.sample(self._sample_user_list, 1)[0]
        # sample a positive event
        ei = random.sample(self._ue.front_list[u], 1)[0]
        # sample a negative event
        while True:
            ej = random.sample(self.events, 1)[0]
            if ej not in self._ue.front_dict[u]:
                break
        # groups
        gi = self._eg.front_list[ei][0]
        gj = self._eg.front_list[ej][0]
        # location
        li = self._el.front_list[ej][0]
        lj = self._el.front_list[ej][0]

        # relationship
        ugi = self._ug.front_dict[u].get(gi, 0)
        ugj = self._ug.front_dict[u].get(gj, 0)
        ug_relation = _sign(ugi - ugj)

        uli = self._ul.front_dict[u].get(li, 0)
        ulj = self._ul.front_dict[u].get(lj, 0)
        ul_relation = _sign(uli - ulj)
        sample = _Sample(u, ei, ej, gi, gj, li, lj, ug_relation, ul_relation)
        return sample

    def batch(self, batch_size=5):
        samples = []
        while True:
            while len(samples) < batch_size:
                samples.append(self._sample())
            # reshape to batch
            samples = _Sample(*list(zip(*samples)))
            yield samples
            samples = []

    def _pos_and_auc_test(self, base, batch):
        res = []
        for ux, ex in base.pairs:
            gx = self._eg.front_list[ex][0]
            lx = self._el.front_list[ex][0]
            res.append((ux, ex, gx, lx))
            if len(res) == batch:
                output = list(zip(*res))
                yield output
                res = []
        if len(res):
            output = list(zip(*res))
            yield output

    def _neg_test(self, batch):
        res = []
        for ux in self._ue.front_dict:
            for ex in self.events:
                if ex not in self._ue.front_dict[ux]:
                    gx = self._eg.front_list[ex][0]
                    lx = self._el.front_list[ex][0]
                    res.append((ux, ex, gx, lx))
                    if len(res) == batch:
                        output = list(zip(*res))
                        yield output
                        res = []
        if len(res):
            output = list(zip(*res))
            yield output

    def test_samples(self, batch, mode=''):
        assert mode in ('pos', 'neg', 'auc')
        if mode == 'pos':
            return self._pos_and_auc_test(self._ue, batch)
        if mode == 'auc':
            return self._pos_and_auc_test(self._neg_ue, batch)
        if mode == 'neg':
            return self._neg_test(batch)


class DatasetReader(object):

    def __init__(self, city_path, neg_num=-1, test_ratio=0.2, trans_thre=2):
        """
        :param city_path: the path of the city dataset
        :param neg_num: the negative sample number for the test evaluation
                each positive event will have neg_num negative events
        :param test_ratio: the ratio of positive samples
        :param trans_thre: the negative threshold of the transition
        """
        self._N = neg_num
        self._ratio = test_ratio
        self._trans_thre = trans_thre
        with open(city_path, 'r', encoding='utf-8') as fp:
            self._info = json.load(fp)

    @staticmethod
    def _read_location(file):
        location_set, t_set = set(), set()
        tl_pairs = []
        print('reading {}'.format(file))
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp:
                tid, lat, lon = line.strip().split(',')
                loc = (float(lat), float(lon))
                location_set.add(loc)
                t_set.add(tid)
                tl_pairs.append((tid, loc))
        return t_set, location_set, tl_pairs

    @staticmethod
    def _filter(pairs, front, tail):
        """filter the pairs by the given front set and tail set"""
        return [(t, l) for t, l in pairs if t in front and l in tail]

    @staticmethod
    def _read_pairs(file):
        f_set, s_set, pairs = set(), set(), []
        print('reading {}'.format(file))
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp:
                f, s = line.strip().split(',')
                f_set.add(f)
                s_set.add(s)
                pairs.append((f, s))
        return f_set, s_set, pairs

    def _split_user_event(self, ue):
        print('split the user-event pairs')
        train_num = int((1 - self._ratio) * len(ue))
        train_ue = ue[:train_num]
        test_ue = ue[train_num:]

        # filter the test user, only keep the user in the train set (due to problem setting)
        test_user = set((ux for ux, _ in test_ue))
        train_user = set((ux for ux, _ in train_ue))
        test_user = test_user & train_user

        # filter the test user-event pair
        test_event = set((ex for _, ex in test_ue))
        test_ue = self._filter(test_ue, test_user, test_event)

        # sample negative pair
        test_ue_pair = _Pairs().fit(test_ue)
        total_ue_pair = _Pairs().fit(ue)

        test_neg_pairs = []
        for uid in test_ue_pair.front_dict:
            # the negative event for the user
            neg_events = test_event - set(total_ue_pair.front_dict.get(uid, []))
            ne_num = self._N * len(test_ue_pair.front_dict[uid])
            if self._N < 0:
                ne_num = len(neg_events)
            if len(neg_events) > ne_num:
                neg_list = random.sample(neg_events, ne_num)
            else:
                neg_list = neg_events
            for nx in neg_list:
                test_neg_pairs.append((uid, nx))

        print('train user-event pair number: {}'.format(len(train_ue)))
        print('test positive user-event pair number: {}'.format(len(test_ue)))
        print('test negative user-event pair number: {}'.format(len(test_neg_pairs)))
        return train_ue, test_ue, test_neg_pairs

    def read(self):
        uset_l, lset_u, ul = self._read_location(self._info['user_location_file'])
        eset_l, lset_e, el = self._read_location(self._info['event_location_file'])

        uset_e, eset_u, ue = self._read_pairs(self._info['ue_file'])
        uset_g, gset_u, ug = self._read_pairs(self._info['ug_file'])
        eset_g, gset_e, eg = self._read_pairs(self._info['eg_file'])

        # base component: total
        users = uset_l & uset_e & uset_g
        events = eset_l & eset_u & eset_l
        groups = gset_e | gset_u
        locations = lset_e | lset_u

        # base paris:
        ue = self._filter(ue, users, events)
        ug = self._filter(ug, users, groups)
        ul = self._filter(ul, users, locations)
        eg = self._filter(eg, events, groups)
        el = self._filter(el, events, locations)

        # divide the user-event pairs
        train_ue, test_ue, test_neg_ue = self._split_user_event(ue)

        train_dataset = Dataset(trans_thre=self._trans_thre).build(train_ue, ug, eg, ul, el, neg_ue=[])
        test_dataset = Dataset(trans_thre=self._trans_thre).build(test_ue, ug, eg, ul, el, neg_ue=test_neg_ue)
        return train_dataset, test_dataset
