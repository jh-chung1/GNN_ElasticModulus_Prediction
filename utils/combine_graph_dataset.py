import numpy as np
import pickle
import os

def combine_graph_data(cover_intervals, overlaps, rock_id, subcube_size, data_num, except_size, save_path):
    """
    Combine graph dataset that from different rocks and subcube size
    """
    rock_names = "_".join(rock_id) if rock_id else "whole"
    if len(except_size) == 1:
        except_sizes = except_size[0]
    else:    
        except_sizes = "_".join(except_size) if except_size else "None"
    
    for cover_interval in cover_intervals:
        for overlap in overlaps:
            combined_data = []

            for Rock in rock_id:
                for i in range(len(subcube_size)):
                    file = f'/dataset_1_to_{data_num[i]}.pkl'
                    data_dir = f'/scratch/users/jhchung1/GNNs/graph_data/geomechanics_data/graph_data/{Rock}/size_{subcube_size[i]}/Mapper_DFS_CoverInterval_{cover_interval}_Overlaps_{overlap}_sol_pore_feature_v3'
                    with open(data_dir + file, 'rb') as f:
                        data = pickle.load(f)
                        combined_data.extend(data)

            save_file = os.path.join(save_path, f'{rock_names}_DFS_CoverInterval_{cover_interval}_Overlap_{overlap}_except_subcube{except_sizes}.pkl')
            print(f'saved: {rock_names}_DFS_CoverInterval_{cover_interval}_Overlap_{overlap}_except_subcube{except_sizes}')

            with open(save_file, 'wb') as f:
                pickle.dump(combined_data, f)
