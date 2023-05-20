import numpy as np
import pandas as pd
import torch

from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_cluster import knn_graph

class MyOwnDataset(Dataset):
    """
    A Dataset object that returns a training sample and its label on the fly.
    """
    
    def __init__(self, idx_batch, path_batch, path_meta, path_sensor, target_mode='angles', K=10, features=['x', 'y', 'z', 'time', 'charge'], threshold_events=500, targets_test=False):
        """
        idx_batch (int): the index in the name "....batch_XXX.parquet". So in [1, 660] for the train set.
        """
        super().__init__(None, transform=None, pre_transform=None)
        
        print(f'Preparing batch {idx_batch} ...')
        
        self.idx_batch = idx_batch
        self.K = K
        self.target_mode = target_mode
        self.threshold_events = threshold_events
        self.features = features
        
        # 1. Load meta data for the batch
        print('Loading meta ...')
        self.meta_df = pd.read_parquet(path_meta)
        self.meta_df = self.meta_df[self.meta_df.batch_id == idx_batch] # slice, keep rows relevant to the batch only
        # TODO: My code currently need targets, even for test. To remove, need to change inference loop first.
        if targets_test:
            self.meta_df['azimuth'] = 0.
            self.meta_df['zenith'] = 0.
        
        # 2. Get sensor coords, centered, normalized
        print('Loading sensor ...')
        self.sensor_geometry = pd.read_csv(path_sensor)
        
        # 3. Load event dataframe
        print('Loading batch ...')
        self.batch_df = pd.read_parquet(path_batch).reset_index()
        
        # 4. Normalize data to help learning.
        self._normalize_data()
        
        # 5. Add sensor info to pulses
        self.batch_df = pd.merge(left=self.batch_df, 
                                 right=self.sensor_geometry,
                                 how='left',
                                 on='sensor_id')
        
        # 6. Set target converter.
        self._set_target_converter()
        
        print('Complete.\n')
        
    def len(self):
        # Because we cropped meta_df, it only contains rows related to this batch.
        # So num rows = num events in batch.
        return len(self.meta_df)

    def get(self, idx_event):
        """
        Gets the i-th event of the batch and its event_id.
        """
        row = self.meta_df.iloc[idx_event]
        df_event = self.batch_df.iloc[row.first_pulse_index.astype(int):row.last_pulse_index.astype(int)+1].copy()
        
        # Events with thousands of pulses create huge graphs => OOM or weird bugs.
        # If too big, we randomly pick a set of nodes within the pulse nodes.
        n_pulses = len(df_event)
        if n_pulses > self.threshold_events:
            df_event = df_event.iloc[np.random.choice(n_pulses, self.threshold_events)]
            n_pulses = self.threshold_events

        # As mentionned in the Data section of the challenge, the time of the pulse in nanoseconds 
        # in the current event time window. The absolute time of a pulse has no relevance, and only 
        # the relative time with respect to other pulses within an event is of relevance.
        df_event.time -= df_event.time.min()
        
        # Convert targets into the desired format
        labels = self._get_targets(row)
        
        # Temporary dataframes - point coords and points features
        df_event_spatial = df_event[['x', 'y', 'z']].to_numpy()
        df_event_features = df_event[self.features].to_numpy()        

        x = torch.tensor(df_event_features, dtype=torch.float)
        pos = torch.tensor(df_event_spatial, dtype=torch.float)
        
        # Create the nodes of the future graph
        data = Data(
            x=x, 
            pos=pos,
            y=labels, 
            n_pulses=torch.tensor(n_pulses, dtype=torch.int32)
            )
        
        # So far `data` is a cloud of points, in 3D. No edges were defined.
        # PyGeometric will create a graph for us, connecting each node
        # to its k nearest neighbors. The example I used as a baseline: 
        # https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing
        # In Graphnet, authors use K=8.
        # Note that if you use DynEdge convolutions then, defining a graph is useless bc the convolution
        # defines new edges at each step.
        # TODO: CV to properly choose k
        # TODO 2: Maybe move it inside models for the reason mentionned above.
        data.edge_index = knn_graph(data.pos, k=self.K)
        
        return data, int(row.event_id)
    
    def _normalize_data(self):
        
        self.sensor_geometry.x -= self.sensor_geometry.x.mean()
        self.sensor_geometry.y -= self.sensor_geometry.y.mean()
        self.sensor_geometry.z -= self.sensor_geometry.z.mean()
        self.sensor_geometry[['x', 'y', 'z']] /= 600 # normalize spatial coords
        
        self.batch_df['time'] = (self.batch_df['time'] - 1.0e04) / 3.0e4 # time
        self.batch_df['charge'] = np.log(self.batch_df['charge']) / 3 # log scale for energy
        self.batch_df['auxiliary'] = self.batch_df['auxiliary'].astype(int) - 0.5
        
        
    def _set_target_converter(self):
        assert self.target_mode in ['angles', 'cossin', 'xyz'], ("Check your target mode.")
        
        if self.target_mode == 'angles':
            self._get_targets = self._targets_angles
        if self.target_mode == 'cossin':
            self._get_targets = self._targets_cossin
        if self.target_mode == 'xyz':
            self._get_targets = self._targets_xyz
    
    def _targets_angles(self, row):
        """Return angles unchanged."""
        return torch.Tensor([row.azimuth, row.zenith])
    
    def _targets_cossin(self, row):
        """Return cos and sin of both angles."""
        # Convert output into a [4] shape tensor
        return torch.Tensor([np.cos(row.azimuth), 
                             np.sin(row.azimuth), 
                             np.cos(row.zenith), 
                             np.sin(row.zenith)])
    
    def _targets_xyz(self, row):
        """Return the 3D coords of the point at this az and zen."""
        return torch.Tensor([np.cos(row.azimuth) * np.sin(row.zenith), 
                             np.sin(row.azimuth) * np.sin(row.zenith),
                             np.cos(row.zenith)])