import os
import pickle
import shutil
from collections import defaultdict
from multiprocessing import Pool
import h5py
import numpy as np
import torch
from metadrive.scenario.scenario_description import MetaDriveType
from scenarionet.common_utils import read_scenario, read_dataset_summary
from torch.utils.data import Dataset
from tqdm import tqdm

from unitraj.datasets import common_utils
from unitraj.datasets.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
from unitraj.datasets.types import object_type, polyline_type
from unitraj.utils.visualization import check_loaded_data
from functools import lru_cache

from unitraj.datasets.base_dataset import BaseDataset

class PersonaDataset(BaseDataset):

    def load_data(self):
        self.data_loaded = {}
        if self.is_validation:
            print('Loading validation data...')
        else:
            print('Loading training data...')

        for cnt, data_path in enumerate(self.data_path):
            phase, dataset_name = data_path.split('/')[-2],data_path.split('/')[-1]
            self.cache_path = os.path.join(self.config['cache_path'], dataset_name, phase)

            data_usage_this_dataset = self.config['max_data_num'][cnt]
            self.starting_frame = self.config['starting_frame'][cnt]
            if self.config['use_cache'] or is_ddp():
                file_list = self.get_data_list(data_usage_this_dataset)
            else:
                if os.path.exists(self.cache_path) and self.config.get('overwrite_cache', False) is False:
                    print('Warning: cache path {} already exists, skip '.format(self.cache_path))
                    file_list = self.get_data_list(data_usage_this_dataset)
                else:

                    _, summary_list, mapping = read_dataset_summary(data_path)

                    if os.path.exists(self.cache_path):
                        shutil.rmtree(self.cache_path)
                    os.makedirs(self.cache_path, exist_ok=True)
                    process_num = os.cpu_count()//2
                    print('Using {} processes to load data...'.format(process_num))

                    is_traverse = True if "traverse" in data_path else False
                    is_traverse = [is_traverse] * process_num

                    data_splits = np.array_split(summary_list, process_num)

                    data_splits = [(data_path, mapping, list(data_splits[i]), dataset_name) for i in range(process_num)]
                    # save the data_splits in a tmp directory
                    os.makedirs('tmp', exist_ok=True)
                    for i in range(process_num):
                        with open(os.path.join('tmp', '{}.pkl'.format(i)), 'wb') as f:
                            pickle.dump(data_splits[i], f)

                    # results = self.process_data_chunk(0)
                    with Pool(processes=process_num) as pool:
                        results = pool.starmap(self.process_data_chunk, zip(list(range(process_num)), is_traverse))

                    # concatenate the results
                    file_list = {}
                    for result in results:
                        file_list.update(result)

                    with open(os.path.join(self.cache_path, 'file_list.pkl'), 'wb') as f:
                        pickle.dump(file_list, f)

                    data_list = list(file_list.items())
                    np.random.shuffle(data_list)
                    if not self.is_validation:
                        # randomly sample data_usage number of data
                        file_list = dict(data_list[:data_usage_this_dataset])

            print('Loaded {} samples from {}'.format(len(file_list), data_path))
            self.data_loaded.update(file_list)

            if self.config['store_data_in_memory']:
                print('Loading data into memory...')
                for data_path in file_list.keys():
                    with open(data_path, 'rb') as f:
                        data = pickle.load(f)
                    self.data_loaded_memory.append(data)
                print('Loaded {} data into memory'.format(len(self.data_loaded_memory)))

        self.data_loaded_keys = list(self.data_loaded.keys())
        print('Data loaded')

    def process_data_chunk(self, worker_index, is_traverse=False):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                # try:
                output = self.preprocess(scenario, is_traverse)

                output = self.process(output)

                output = self.postprocess(output)

                # except Exception as e:
                #     print('Warning: {} in {}'.format(e, file_name))
                #     output = None

                if output is None: continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    for key, value in record.items():
                        if isinstance(value, str):
                            value = np.bytes_(value)
                        grp.create_dataset(key, data=value)
                    file_info = {}
                    kalman_difficulty = np.stack([x['kalman_difficulty'] for x in output])
                    file_info['kalman_difficulty'] = kalman_difficulty
                    file_info['h5_path'] = hdf5_path
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list
    
    def preprocess(self, scenario, is_traverse=False):
        traffic_lights = scenario['dynamic_map_states']
        tracks = scenario['tracks']
        map_feat = scenario['map_features']

        past_length = self.config['past_len']
        future_length = self.config['future_len']
        total_steps = past_length + future_length
        starting_fame = self.starting_frame
        ending_fame = starting_fame + total_steps
        trajectory_sample_interval = self.config['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)

        track_infos = {
            'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
            'object_type': [],
            'trajs': []
        }

        for k, v in tracks.items():

            state = v['state']
            for key, value in state.items():
                if len(value.shape) == 1:
                    state[key] = np.expand_dims(value, axis=-1)
            all_state = [state['position'], state['length'], state['width'], state['height'], state['heading'],
                         state['velocity'], state['valid']]
            # type, x,y,z,l,w,h,heading,vx,vy,valid
            all_state = np.concatenate(all_state, axis=-1)
            # all_state = all_state[::sample_inverval]
            if all_state.shape[0] < ending_fame:
                all_state = np.pad(all_state, ((ending_fame - all_state.shape[0], 0), (0, 0)))
            all_state = all_state[starting_fame:ending_fame]

            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'

            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        # scenario['metadata']['ts'] = scenario['metadata']['ts'][::sample_inverval]
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]

        # x,y,z,type
        map_infos = {
            'lane': [],
            'road_line': [],
            'road_edge': [],
            'stop_sign': [],
            'crosswalk': [],
            'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue

            cur_info = {'id': k}
            cur_info['type'] = v['type']
            if polyline_type_ in [1, 2, 3]:
                cur_info['speed_limit_mph'] = v.get('speed_limit_mph', None)
                cur_info['interpolating'] = v.get('interpolating', None)
                cur_info['entry_lanes'] = v.get('entry_lanes', None)
                try:
                    cur_info['left_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['left_neighbor']
                    ]
                    cur_info['right_boundary'] = [{
                        'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                        'feature_id': x['feature_id'],
                        'boundary_type': 'UNKNOWN'  # roadline type
                    } for x in v['right_neighbor']
                    ]
                except:
                    cur_info['left_boundary'] = []
                    cur_info['right_boundary'] = []
                polyline = np.array(v['polyline'])
                polyline = interpolate_polyline(polyline)
                map_infos['lane'].append(cur_info)
            elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
                try:
                    polyline = np.array(v['polyline'])
                except:
                    polyline = np.array(v['polygon'])
                polyline = interpolate_polyline(polyline)
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [15, 16]:
                polyline = np.array(v['polyline'])
                polyline = interpolate_polyline(polyline)
                cur_info['type'] = 7
                map_infos['road_line'].append(cur_info)
            elif polyline_type_ in [17]:
                cur_info['lane_ids'] = v['lane']
                cur_info['position'] = v['position']
                map_infos['stop_sign'].append(cur_info)
                polyline = np.array(v['position'])[np.newaxis]
            elif polyline_type_ in [18]:
                map_infos['crosswalk'].append(cur_info)
                polyline = np.array(v['polygon'])
            elif polyline_type_ in [19]:
                map_infos['crosswalk'].append(cur_info)
                polyline = np.array(v['polygon'])
            if polyline.shape[-1] == 2:
                polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
            try:
                cur_polyline_dir = get_polyline_dir(polyline)
                type_array = np.zeros([polyline.shape[0], 1])
                type_array[:] = polyline_type_
                cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
            except:
                cur_polyline = np.zeros((0, 7), dtype=np.float32)
            polylines.append(cur_polyline)
            cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos['all_polylines'] = polylines

        dynamic_map_infos = {
            'lane_id': [],
            'state': [],
            'stop_point': []
        }
        for k, v in traffic_lights.items():  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:  # (num_observed_signals)
                lane_id.append(str(v['lane']))
                state.append(cur_signal)
                if type(v['stop_point']) == list:
                    stop_point.append(v['stop_point'])
                else:
                    stop_point.append(v['stop_point'].tolist())
            # lane_id = lane_id[::sample_inverval]
            # state = state[::sample_inverval]
            # stop_point = stop_point[::sample_inverval]
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))
            dynamic_map_infos['state'].append(np.array([state]))
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))

        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        ret.update(scenario['metadata'])
        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.config['past_len'] - 1
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])

        try:
            if self.config['only_train_on_ego'] or is_traverse:
                tracks_to_predict = {
                    'track_index': [ret['sdc_track_index']],
                    'difficulty': [0],
                    'object_type': [MetaDriveType.VEHICLE]
                }
            elif ret.get('tracks_to_predict', None) is None:
                filtered_tracks = self.trajectory_filter(ret)
                sample_list = list(filtered_tracks.keys())
                tracks_to_predict = {
                    'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                    id in track_infos['object_id']],
                    'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                    id in track_infos['object_id']],
                }
            else:
                sample_list = list(ret['tracks_to_predict'].keys())  # + ret.get('objects_of_interest', [])
                sample_list = list(set(sample_list))
                tracks_to_predict = {
                    'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                    id in track_infos['object_id']],
                    'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                    id in track_infos['object_id']],
                }
        except Exception as _:
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }

        ret['tracks_to_predict'] = tracks_to_predict
        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]
        ret['track_length'] = total_steps
        return ret