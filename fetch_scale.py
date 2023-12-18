"""ALEX: helper script to load the sclaing.

python fetch_scale.py --dir /mnt/fischer/alex/causalsim --left_out_policy linear_bba
"""

import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir", help="root directory")
    parser.add_argument("--left_out_policy", type=str, help="left out policy")
    args = parser.parse_args()

    data_dir = f'{args.dir}subset_data/{args.left_out_policy}'
    buff_mean = np.load(f'{data_dir}/buffs_mean.npy')
    buff_std = np.load(f'{data_dir}/buffs_std.npy')
    next_buff_mean = np.load(f'{data_dir}/next_buffs_mean.npy')
    next_buff_std = np.load(f'{data_dir}/next_buffs_std.npy')
    action_mean = np.load(f'{data_dir}/actions_mean.npy')
    action_std = np.load(f'{data_dir}/actions_std.npy')
    trans_time_mean = np.load(f'{data_dir}/dts_mean.npy')
    trans_time_std = np.load(f'{data_dir}/dts_std.npy')
    # chat is c_hat is throughput
    throughput_mean = np.load(f'{data_dir}/chats_mean.npy')
    throughput_std = np.load(f'{data_dir}/chats_std.npy')

    combined = pd.DataFrame({
        'metric': ['mean', 'std'],
        'buffer': [buff_mean, buff_std],
        'next_buffer': [next_buff_mean, next_buff_std],
        'size': [action_mean, action_std],
        'throughput': [throughput_mean, throughput_std],
        'trans_time': [trans_time_mean, trans_time_std],
    })
    print(combined.to_csv(index=False))
