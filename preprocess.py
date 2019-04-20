import os
import json
import math


class Extractor(object):
    """
    extract the dataset of a city from the origin dataset
    """

    def __init__(self, save_base='.', radius=25, wash=3, user_file='user_lon_lat.csv', event_file='event_lon_lat.csv',
                 ue='user_event.csv', ug='user_group.csv', eg='event_group.csv'):
        """
        :param radius: the kilometers from the center of the city
        """
        self._save_base = save_base
        self.radius = radius
        self.R = 6371.004
        self.wash = wash

        self.user_name = user_file
        self.event_name = event_file
        self.user_event_name = ue
        self.user_group_name = ug
        self.event_group_name = eg

        self.user_list = {}
        self.event_list = {}
        self.group_list = {}

    def _distance(self, location, center):
        """compute the distance from the center to a location"""
        lon = math.radians(location[0])
        lat = math.radians(location[1])
        a = lat - math.radians(center[1])
        b = lon - math.radians(center[0])
        ss = math.sqrt(math.sin(a / 2) ** 2 +
                       math.cos(lat) * math.cos(center[0]) *
                       (math.sin(b / 2) ** 2))
        ss = max(-1.0, min(1.0, ss))
        s = 2 * math.asin(ss)
        return s * self.R

    def _read_location(self, filename, center):
        terms = set()
        print('reading location `{}`'.format(filename))
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                tid, lon, lat = line.strip().split(',')
                lon = float(lon)
                lat = float(lat)
                if self._distance([lon, lat], center) < self.radius:
                    terms.add(tid)
        return terms

    @staticmethod
    def _item_set_from_pairs(filename):
        fset, sset = set(), set()
        print('reading pair file `{}`'.format(filename))
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                first, second = line.strip().split(',')
                fset.add(first)
                sset.add(second)
        return fset, sset

    @staticmethod
    def _filter_paris(filename, first_set, second_set):
        pairs = []
        print('filtering pairs `{}`'.format(filename))
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                first, second = line.strip().split(',')
                if first in first_set and second in second_set:
                    pairs.append((first, second))
        return pairs

    def _filter_user_with_wash(self):
        user_index = {}
        print('filtering user with `wash={}`'.format(self.wash))
        with open(self.user_event_name, 'r', encoding='utf-8') as fp:
            for line in fp:
                uid, eid = line.strip().split(',')
                if uid not in user_index:
                    user_index[uid] = set()
                user_index[uid].add(eid)
        user_set = set([uid for uid in user_index if len(user_index[uid]) >= self.wash])
        return user_set

    def read(self, center):
        user_set_loc = self._read_location(self.user_name, center)
        event_set_loc = self._read_location(self.event_name, center)

        user_set_ue, event_set_ue = self._item_set_from_pairs(self.user_event_name)
        user_set_ug, group_set_ug = self._item_set_from_pairs(self.user_group_name)
        event_set_eg, group_set_eg = self._item_set_from_pairs(self.event_group_name)

        user_set_wash = self._filter_user_with_wash()

        self._user_set = user_set_loc & user_set_ue & user_set_ug & user_set_wash
        self._event_set = event_set_loc & event_set_eg & event_set_ue
        self._group_set = group_set_eg & group_set_ug
        print('user', len(self._user_set))
        print('event', len(self._event_set))
        print('groups', len(self._group_set))

        self._ue = self._filter_paris(self.user_event_name, self._user_set, self._event_set)
        self._ug = self._filter_paris(self.user_group_name, self._user_set, self._group_set)
        self._eg = self._filter_paris(self.event_group_name, self._event_set, self._group_set)
        return self

    def _save_info(self, city_name):
        base = os.path.join(self._save_base, city_name)
        if not os.path.exists(base):
            os.makedirs(base)

        filename = os.path.join(self._save_base, city_name, 'info.json')
        data = {
            'user_location_file': os.path.join(self._save_base, city_name, 'user.loc'),
            'event_location_file': os.path.join(self._save_base, city_name, 'event.loc'),
            'ue_file': os.path.join(self._save_base, city_name, 'ue.pair'),
            'ug_file': os.path.join(self._save_base, city_name, 'ug.pair'),
            'eg_file': os.path.join(self._save_base, city_name, 'eg.pair'),
        }
        with open(filename, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)
        return data

    @staticmethod
    def _save_location(in_file, out_file, id_set):
        print('saving `{}`'.format(out_file))
        with open(in_file, 'r', encoding='utf-8') as fin, open(out_file, 'w', encoding='utf-8') as fout:
            for line in fin:
                tid, lat, lon = line.strip().split(',')
                if tid in id_set:
                    fout.write(line)

    @staticmethod
    def _save_paris(filename, pairs):
        print('saving `{}`'.format(filename))
        with open(filename, 'w', encoding='utf-8') as fp:
            for fid, sid in pairs:
                fp.write('{},{}\n'.format(fid, sid))

    def save(self, city_name):
        info = self._save_info(city_name)
        self._save_location(self.user_name, info['user_location_file'], self._user_set)
        self._save_location(self.event_name, info['event_location_file'], self._event_set)
        self._save_paris(info['ue_file'], self._ue)
        self._save_paris(info['ug_file'], self._ug)
        self._save_paris(info['eg_file'], self._eg)
        return self

    def __call__(self, city_name, center):
        self.read(center)
        self.save(city_name)


def preprocess(config):
    cities = [
        # 'San Francisco',
        # 'New York',
        # 'Los Angeles',
        'Houston',
        # 'Chicago',
        # 'Washington',
        # 'Phoenix',
        # 'San Jose'
    ]
    loc = {
        'San Francisco': [-122.41999816894531, 37.779998779296875],
        'New York': [-73.98999786376953, 40.75],
        'Los Angeles': [-118.23999786376953, 33.970001220703125],
        'Houston': [-95.22000122070312, 29.719999313354492],
        'Chicago': [-87.62000274658203, 41.880001068115234],
        'Washington': [-77.0199966430664, 38.90999984741211],
        'Phoenix': [-112.06999969482422, 33.45000076293945],
        'San Jose': [-121.9000015258789, 37.38999938964844]
    }

    extractor = Extractor(config['save_base'], config['radius'], config['wash'],
                          config['user_file'], config['event_file'],
                          config['ue_file'], config['ug_file'], config['eg_file'])
    for city in cities:
        extractor(city, loc[city])
