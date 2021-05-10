import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import random


# Hyper Parameters
DATASET_PATH = ''
USER_CNT = 6040
ITEM_CNT = 3416
MAX_POS = 200

RANDOM_MASK_CNT = 16
MASK_RATE = 0.2
SINGLE_MASK_RATE = 0.2


# Data Class
class Rate_Info:
    def __init__(self, user_id, item_id, rate_value, rate_time):
        self.user_id: int = user_id
        self.item_id: int = item_id
        self.rate_value: float = rate_value
        self.rate_time: float = rate_time


# Global Variables
rate_info = []
rate_matrix = None
user_matrix = None
user_rate_cnt = None

user_latent = None
item_latent = None

# Item Popular
item_popular = None

# Train, Valid, Eval ID
train_id, valid_id, test_id = None, None, None
time_split = True


# Load the data from file.
def load_from_file():
    print('Loading data...')
    fin = open('datasets/' + DATASET_PATH + '.txt', 'rt')
    for line in fin.readlines():
        user_id, item_id = line.split(' ')
        user_id = int(user_id) - 1
        item_id = int(item_id)
        rate_value = 1  # int(rate_value)
        rate_time = 1  # int(rate_time)
        rate_info.append(Rate_Info(user_id, item_id, rate_value, rate_time))
    print('Finished.')


# Load Data
def load_data(hyper_params: dict):
    print('Start loading the data...')

    global ITEM_CNT, USER_CNT, DATASET_PATH, MAX_POS, item_popular, rate_info
    global train_id, valid_id, test_id, time_split
    ITEM_CNT = hyper_params['total_items']
    USER_CNT = hyper_params['total_users']
    DATASET_PATH = hyper_params['dataset_path']
    MAX_POS = hyper_params['seq_len']

    item_popular = torch.zeros((ITEM_CNT + 2), dtype=torch.float)

    # Initialize User_Matrix.
    global user_rate_cnt, user_matrix
    if hyper_params.get('time_split') is not None:
        time_split = hyper_params['time_split']

    if time_split:
        pkl_path = './model_dat/time_split_' + DATASET_PATH + '.pkl'
    else:
        pkl_path = './model_dat/user_split_' + DATASET_PATH + '.pkl'

    try:
        user_rate_cnt, user_matrix, item_popular, train_id, valid_id, test_id = torch.load(
            pkl_path)
    except Exception as e:
        print(e)
        print('Load from file failed.')

        # Load the data from the datasets files.
        load_from_file()

        user_matrix = torch.zeros((USER_CNT, MAX_POS + 1), dtype=torch.long)
        user_rate_cnt = torch.zeros(USER_CNT, dtype=torch.long)

        # Sort the rate items by time.
        # rate_info = sorted(rate_info, key=lambda x: x.rate_time, reverse=True)
        rate_info.reverse()

        print('Loading into tensor...')

        rate_info_len = len(rate_info)
        for i, rate in enumerate(rate_info):
            # Write into the item popular.
            item_popular[rate.item_id] += 1.0

            if user_rate_cnt[rate.user_id] < MAX_POS + 1:
                user_rate_cnt[rate.user_id] += 1
                user_matrix[rate.user_id, -user_rate_cnt[rate.user_id]
                            ] = rate.item_id

            if i % 20000 == 0:
                print(f'{i}/{rate_info_len} finished.')

        # Softmax the item popular.
        item_popular[0] = 0.0

        if not time_split:
            hold_out_cnt = int(hyper_params['hold_out_prop'] * USER_CNT)
            total_id = np.array(range(USER_CNT))
            np.random.shuffle(total_id)
            total_id = total_id.tolist()
            train_id = total_id[:USER_CNT - 2 * hold_out_cnt]
            valid_id = total_id[USER_CNT - 2 *
                                hold_out_cnt:USER_CNT - hold_out_cnt]
            test_id = total_id[USER_CNT - hold_out_cnt:]
        else:
            train_id = list(range(USER_CNT))
            valid_id = list(range(USER_CNT))
            test_id = list(range(USER_CNT))

        # Save the pkl.
        torch.save((user_rate_cnt, user_matrix, item_popular, train_id, valid_id, test_id),
                   pkl_path)

    print('Data loaded successfully.')


# Generate A Single Data
def generate_single_train_data(user_id):
    # Random user id.
    cur_user_id = user_id
    cur_cnt = user_rate_cnt[cur_user_id].item()
    cur_src = user_matrix[cur_user_id, -cur_cnt:]

    cur_data_x = torch.zeros(
        [MAX_POS], dtype=torch.long)
    cur_padding = torch.zeros([MAX_POS], dtype=torch.bool)
    cur_data_y = torch.zeros([MAX_POS], dtype=torch.long)
    cur_padding = torch.cat((torch.zeros(
        cur_cnt - 1), torch.ones(MAX_POS - cur_cnt + 1)), dim=0).bool()

    cur_data_x[:cur_cnt - 1] = cur_src[:cur_cnt - 1]
    cur_data_y[:cur_cnt - 1] = cur_src[1:cur_cnt]

    return cur_data_x, cur_data_y, cur_padding, cur_user_id, cur_cnt - 1


# Generate single train data with split
def generate_single_train_data_split(hyper_params, user_id):
    cur_cnt = user_rate_cnt[user_id].item()
    test_cnt = int(hyper_params['test_prop'] * cur_cnt)
    train_cnt = cur_cnt - test_cnt
    cur_src = user_matrix[user_id, -cur_cnt: - test_cnt]

    cur_data_x = torch.zeros([MAX_POS], dtype=torch.long)
    cur_data_y = torch.zeros([MAX_POS], dtype=torch.long)
    cur_padding = torch.cat(
        (torch.zeros(train_cnt - 1), torch.ones(MAX_POS - train_cnt + 1)), dim=0).bool()

    cur_data_x[:train_cnt - 1] = cur_src[:train_cnt - 1]
    cur_data_y[:train_cnt - 1] = cur_src[1:train_cnt]

    return cur_data_x, cur_data_y, cur_padding, user_id, train_cnt - 1


def generate_single_eval_data(hyper_params, user_id):
    cur_cnt = user_rate_cnt[user_id].item()
    test_cnt = int(hyper_params['test_prop'] * cur_cnt)
    train_cnt = cur_cnt - test_cnt
    cur_src = user_matrix[user_id, -cur_cnt:]

    cur_data_x = torch.zeros([MAX_POS], dtype=torch.long)
    cur_data_x[:train_cnt] = cur_src[:train_cnt]
    cur_data_y = torch.zeros([MAX_POS], dtype=torch.long)
    cur_data_y[:test_cnt] = cur_src[train_cnt:]
    cur_padding = torch.cat((torch.zeros(
        train_cnt), torch.ones(MAX_POS - train_cnt)), dim=0).bool()
    cur_cnt = train_cnt

    return cur_data_x, cur_data_y, cur_padding, user_id, train_cnt


def generate_index_data(hyper_params, index: list, is_train=True):
    train_data_x = torch.zeros([len(index), MAX_POS], dtype=torch.long)
    train_data_y = torch.zeros([len(index), MAX_POS], dtype=torch.long)
    train_padding = torch.zeros([len(index), MAX_POS], dtype=torch.bool)
    train_user_id = torch.zeros([len(index)], dtype=torch.long)
    train_cur_cnt = torch.zeros([len(index)], dtype=torch.long)

    for i, user_id in enumerate(index):
        if is_train:
            if time_split:
                cur_train_data = generate_single_train_data_split(
                    hyper_params, user_id)
            else:
                cur_train_data = generate_single_train_data(user_id)
        else:
            cur_train_data = generate_single_eval_data(hyper_params, user_id)
        train_data_x[i] = cur_train_data[0]
        train_data_y[i] = cur_train_data[1]
        train_padding[i] = cur_train_data[2]
        train_user_id[i] = cur_train_data[3]
        train_cur_cnt[i] = cur_train_data[4]

    return torch.utils.data.TensorDataset(train_data_x, train_data_y, train_padding, train_user_id, train_cur_cnt)


def generate_train_data(hyper_params):
    print('Generating train data...')
    dataset = generate_index_data(hyper_params, index=train_id, is_train=True)
    print('Test data generate succesfully.')
    return dataset


def generate_test_data(hyper_params):
    print('Generating test data...')
    dataset = generate_index_data(hyper_params, index=test_id, is_train=False)
    print('Test data generate successfully.')
    return dataset


def generate_validate_data(hyper_params):
    print('Generating validate data...')
    dataset = generate_index_data(hyper_params, index=valid_id, is_train=False)
    print('Validate data generate successfully.')
    return dataset


if __name__ == '__main__':
    hyper_params = {
        'total_items': 20720,
        'total_users': 136677,
        'seq_len': 200,
        'dataset_path': 'ml-20m',
        'kl_weight': 0.05,
        'contrast_weight': 0.1,
        'item_embed_size': 128,
        'rnn_size': 100,
        'hidden_size': 100,
        'latent_size': 64,
        'timesteps': 5,
        'hold_out_prop': 0.125,
        'test_prop': 0.2,
        'batch_size': 64
    }

    load_data(hyper_params)
