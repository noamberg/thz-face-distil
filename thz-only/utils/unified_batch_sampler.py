#!/usr/bin/env python3
"""
MTL Balanced Batch Sampler
Ensures each batch contains a perfectly balanced set of samples:
- N identities (e.g., 20)  
- M postures (e.g., 5)
Total batch size = N * M (e.g., 100)

For each batch, it yields one sample for every (identity, posture) combination.
"""

import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Sampler

def parse_path_info(path_str: str):
    """Extracts the numeric ID and posture from a file path."""
    try:
        path = Path(path_str)
        # Identity ID is the first directory part (e.g., "1", "2", etc.)
        identity_id = path.parts[0]
        # Posture is the second part of the filename after splitting by underscore
        filename_parts = path.stem.split('_')
        posture = filename_parts[1]  # Should be like "b1b", "n1n", etc.
        return identity_id, posture
    except (IndexError, AttributeError):
        return None, None

class MTLBalancedBatchSampler(Sampler):
    """
    Ensures each batch contains a perfectly balanced set of samples:
    - N identities (e.g., 20)
    - M postures (e.g., 5)
    Total batch size = N * M (e.g., 100).

    For each batch, it yields one sample for every (identity, posture) combination.
    """
    def __init__(self, dataframe: pd.DataFrame, num_identities: int = 20, num_postures: int = 5):
        # We do not call super().__init__() as we work directly with indices
        self.dataframe = dataframe
        self.num_identities = num_identities
        self.postures = [f'b{i}b' for i in range(1, num_postures + 1)]
        self.batch_size = self.num_identities * len(self.postures)

        # 1. Group all data indices by (identity, posture)
        self.grouped_indices = self._group_indices()

        # 2. Filter for identities that have data for all required postures
        self.valid_identities = self._get_valid_identities()
        if len(self.valid_identities) < self.num_identities:
            raise ValueError(
                f"Not enough identities with all {len(self.postures)} postures. "
                f"Found {len(self.valid_identities)}, but require {self.num_identities}."
            )
        # Use a fixed set of identities for consistent training epochs
        self.identities_in_use = sorted(self.valid_identities)[:self.num_identities]

        # 3. Determine the number of batches possible
        self.num_batches = self._calculate_num_batches()

        print("MTLBalancedBatchSampler Initialized:")
        print(f"  - Identities in use: {self.num_identities} ({self.identities_in_use})")
        print(f"  - Postures per identity: {len(self.postures)}")
        print(f"  - Batch Size: {self.batch_size}")
        print(f"  - Batches per epoch: {self.num_batches}")

    def _group_indices(self):
        grouped = defaultdict(lambda: defaultdict(list))
        temp_df = self.dataframe.copy().reset_index(drop=True)  # Reset index to match dataset

        parsed_info = temp_df['anchor'].apply(lambda x: pd.Series(parse_path_info(x)))
        temp_df[['identity', 'posture']] = parsed_info

        for idx, row in temp_df.iterrows():
            if row['identity'] and row['posture']:
                grouped[row['identity']][row['posture']].append(idx)
        return grouped

    def _get_valid_identities(self):
        valid_identities = []
        for identity, postures in self.grouped_indices.items():
            if all(p in postures for p in self.postures):
                valid_identities.append(identity)
        return valid_identities

    def _calculate_num_batches(self):
        """The number of batches is limited by the identity/posture with the fewest samples."""
        min_samples = float('inf')
        for identity in self.identities_in_use:
            for posture in self.postures:
                min_samples = min(min_samples, len(self.grouped_indices[identity][posture]))
        return min_samples

    def __iter__(self):
        # Create a temporary, shuffled copy of the indices for this epoch
        shuffled_groups = {
            identity: {
                posture: random.sample(indices, len(indices))
                for posture, indices in self.grouped_indices[identity].items()
            } for identity in self.identities_in_use
        }

        # Yield batches one by one
        for i in range(self.num_batches):
            batch_indices = []
            for identity in self.identities_in_use:
                for posture in self.postures:
                    # Pick the i-th sample from the shuffled list for this group
                    sample_idx = shuffled_groups[identity][posture][i]
                    batch_indices.append(sample_idx)

            # Shuffle the final batch to ensure random order within the batch
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches