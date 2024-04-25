from collections import OrderedDict
import itertools
from dataclasses import dataclass
import pandas as pd


class DataBalancer:
    def __init__(self, seed: int = 13, verbose: bool = False):
        # set the seed
        self.seed = seed

        # initialize variables as none
        self.reference_df = None
        self.target_df = None
        self.feature_list = None
        self.target_ratio = None
        self.level_dict = None
        self.slice_list = None
        self.verbose = verbose

    def set_reference(self, reference_df):
        self.reference_df = reference_df
        dataframe_stats(self.reference_df, "reference dataframe:")

    def set_target(self, target_df):
        self.target_df = target_df
        dataframe_stats(self.target_df, "target dataframe:")

    def set_feature_list(self, feature_list):
        self.feature_list = feature_list
        print(f"balancing distribution based on {', '.join(self.feature_list)}\n")

    def set_balance(self, target_ratio, reference_ratio=1):
        # ref and target ratio should be equal for 50/50 balancing
        # normalize the target ratio so the reference is treated as 1
        # and the ratio is 1:target_ratio 
        self.target_ratio = target_ratio / reference_ratio
        print(f"balancing data with 1.00 : {self.target_ratio:.2f} ratio\n")

    def get_unique_levels(self):
        assert self.reference_df is not None, "please select a reference dataframe with self.set_reference() first !"
        assert self.feature_list is not None, "please select features with self.set_feature_list() first !"

        level_dict = OrderedDict()
        for feature in self.feature_list:
            level_dict[feature] = list(self.reference_df[feature].unique())

        self.level_dict = level_dict

        print(self.level_dict)

    def measure_slices(self):
        assert self.level_dict is not None, "please run self.get_unique_levels() first !"

        self.slice_list = list()

        for feature_levels in itertools.product(*self.level_dict.values()):
            feature_level_dict=dict(zip(list(self.level_dict.keys()), feature_levels))

            # format the current feature level dict into a query and subset the reference df with it
            data_query = ' and '.join(["{} == '{}'".format(k,v) for k,v in feature_level_dict.items()])   
            # then get its length and save it as the reference n in the current DataSlice 
            reference_n = len(self.reference_df.query(data_query))

            self.slice_list.append(DataSlice(
                feature_dict=feature_level_dict,
                reference_n=reference_n
            ))

        print(self.slice_list)
        
    def coerce_to_strings(self):
        assert self.target_df is not None
        assert self.reference_df is not None
        assert self.feature_list is not None
        
        for df in [self.target_df, self.reference_df]:
            for feature in self.feature_list:
                df[feature] = df[feature].astype(str)

    def sample_slices(self):
        assert self.slice_list is not None, "please run self.measure_slices() first !"
        assert self.target_df is not None, "please select a target dataframe with self.set_target() first !"
        
        # coerce all feature cols to strings
        self.coerce_to_strings()

        for data_slice in self.slice_list:
            # format the current feature dict into a query and subset the target df with it
            data_query = ' and '.join(["{} == '{}'".format(k,v) for k,v in data_slice.feature_dict.items()]) 
            # then take a sample with 
            target_subset_df = self.target_df.query(data_query)
            
            # get the reference n and correct it if it's bigger than
            ref_n = data_slice.reference_n
            if ref_n > len(target_subset_df):
                ref_n = len(target_subset_df)
                            
            # if either our reference n or subset length is less than 1 skip it
            if ref_n < 1:
                continue
            
            # sample the ref_n multiplied by the target_ratio
            data_slice.target_slice = target_subset_df.sample(
                int(ref_n * self.target_ratio), 
                random_state=self.seed
            )
            data_slice.target_n = len(data_slice.target_slice)

        print(self.slice_list)

    def merge_slices(self):
        # filter out data slices without a dataframe
        valid_slices = [data_slice for data_slice in self.slice_list if data_slice.target_slice is not None]
        assert len(valid_slices) > 0, "please run self.sample_slices() first !"

        # merge each of the target slices
        balanced_target_df = pd.concat([data_slice.target_slice for data_slice in valid_slices], axis=0)
        return balanced_target_df

            


@dataclass
class DataSlice:
    feature_dict: dict
    reference_n: int
    target_n: int = 0
    target_slice: pd.DataFrame or None = None

    def __repr__(self):
        return f"DataSlice(feature_dict: {self.feature_dict}, reference_n: {self.reference_n}, target_n: {self.target_n})"
    

def dataframe_stats(df, title: str or None = None):
    if title is not None:
        print(f"{title}")
        
    num_patients = df.empi_anon.nunique()
    num_exams = df.acc_anon.nunique()
    
    print(f"Patients: {num_patients:,}")
    print(f"Exams: {num_exams:,}")
    
    if 'png_path' in df.columns:
        print(f"Images: {len(df):,}")
        
    print()



"""
sample usage:

balancer = DataBalancer()

balancer.set_reference(pos_df)
balancer.set_target(neg_df)

# select features
feature_list = ['tissueden', 'ViewPosition', 'Ethnicity', 'Race']
balancer.set_feature_list(feature_list)

# set a 1:1 balance
balancer.set_balance(1, 1)
balancer.get_unique_levels()
balancer.measure_slices()
balancer.sample_slices()
balanced_neg_df = balancer.merge_slices()



"""