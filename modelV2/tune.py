import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import json
import logging
import os
from modelV2.data import ImpromptuDataset, get_augment_list
from modelV2.balancer import DataBalancer
# from dataclasses import dataclass

class HyperParamOptimizer:
    def __init__(self, hparam_range_dict: dict, balance: bool, 
                 monitor_metric: str = 'loss', n_bootstraps: int = 5,
                 epochs_per_run: int = 30, seed: int = 13):
        # load model function and hyperparameter ranges
        self.hparam_range_dict = hparam_range_dict
        self.balance = balance
        self.val_monitor_metric = f"val_{monitor_metric}"
        self.test_monitor_metric = f"test_{monitor_metric}"
        self.n_bootstraps = n_bootstraps
        self.epochs_per_run = epochs_per_run
        self.seed = seed
        
        # initialize parameters
        self.device = None
        # self.loader_dict = None
        self.model_name = None
        self.model_type = None
        self.objective_function = None

        self.df_pool_dict = None
        self.mask_factor = None
        self.batch_size = None
        self.transform = None
        self.bootstrap_prop = None

        self.run_num = 0
        self.boot_num = 0

        # df_pool_dict should look like:
        # {
        #     "pos": {
        #         "train": train_pos_df,
        #         "val": val_pos_df,
        #         "test": test_pos_df
        #     },
        #     "neg": {
        #         "train": train_neg_df,
        #         "val": val_neg_df,
        #         "test": test_neg_df
        #     }
        # }
        
        
    def load_data(self, df_pool_dict: dict, batch_size: int, augment_dict: dict, bootstrap_prop: float = 0.7):
        self.bootstrap_prop = bootstrap_prop

        self.df_pool_dict = df_pool_dict
        self.batch_size = batch_size
        self.transform = get_augment_list(augment_dict)
        print("Data loaded...")
        
    def set_model(self, device, model_type):        
        self.device = device
        self.model_type = model_type
                
    def optimize(self, objective, n_random: int, n_guided: int, model_name: str):
        # set the seed at the start of the optimization so each sequential 
        # generation is reproducible
        np.random.seed(self.seed)

        # set objective function
        self.objective_function = objective
        
        log_path = f"./logs/{model_name}_opt.json"
        self.model_name = model_name
        
        # start the logger
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(message)s')
        
        print(f"\nOptimizing model {'-'*60}")
        print(f"Logging results to '{log_path}'")

        # define optimizer
        optimizer = BayesianOptimization(
            f=self.objective_wrapper,
            pbounds=self.hparam_range_dict,
            random_state=self.seed
        )

        # maximize f1 with hyperparams
        optimizer.maximize(init_points=n_random, n_iter=n_guided)
            
        # open the log file
        opt_df = pd.read_json(log_path, lines=True)
        
        # extract the best performing parameters based on the monitor metric
        best_idx = opt_df[self.test_monitor_metric].idxmax()
        best_param_dict = opt_df.loc[best_idx, 'params']
        
        # train a model and evaluate it on the test set with the best performing parameters
        print('Run using best params: ', best_param_dict)        
        metrics_dict = self.objective_function(
            model_func=self.model_func, 
            device=self.device, 
            loader_dict=self.loader_dict, 
            n_epochs=self.epochs_per_run, 
            save_dir=f"temp_{self.model_name}/", 
            monitor_metric=self.val_monitor_metric,
            seed=self.seed,
            **best_param_dict
        )
        
        log_data(best_param_dict, metrics_dict.test, final=True)

    def _get_bootstrap_dataloader(self, seed, img_col: str = '__target_path'):
        # calculate the set size based on the chosen bootstrap proportion
        train_n = int(self.bootstrap_prop * len(self.df_pool_dict['pos']['train']))
        val_n = int(self.bootstrap_prop * len(self.df_pool_dict['pos']['val']))

        pos_train = self.df_pool_dict['pos']['train'].sample(train_n, random_state=seed)
        pos_val = self.df_pool_dict['pos']['val'].sample(val_n, random_state=seed)

        if self.balance:
            neg_train = self._balance_negative(pos_train, self.df_pool_dict['neg']['train'])
            neg_val = self._balance_negative(pos_val, self.df_pool_dict['neg']['val'])

        else:
            neg_train = self.df_pool_dict['neg']['train'].sample(train_n, random_state=seed)
            neg_val = self.df_pool_dict['neg']['val'].sample(val_n, random_state=seed)

        # use the full pos/neg test sets we provide
        pos_test = self.df_pool_dict['pos']['test']
        neg_test = self.df_pool_dict['neg']['test']

        # log dataframes to track their distributions
        df_dict = {
            "pos": {
                "train": pos_train,
                "val": pos_val,
                "test": pos_test
            },
            "neg": {
                "train": neg_train,
                "val": neg_val,
                "test": neg_test
            }
        }
        self.log_demographics(df_dict)

        train_ds = ImpromptuDataset(
            {'pos':pos_train, 'neg':neg_train}, 
            img_col=img_col, 
            transform=self.transform
        )
        val_ds = ImpromptuDataset(
            {'pos':pos_val, 'neg':neg_val}, 
            img_col=img_col, 
            transform=self.transform
        )
        test_ds = ImpromptuDataset(
            {'pos':pos_test, 'neg':neg_test}, 
            img_col=img_col, 
            transform=self.transform
        )

        train_loader = train_ds.to_loader(self.batch_size)
        val_loader = val_ds.to_loader(self.batch_size)
        test_loader = test_ds.to_loader(self.batch_size)

        loader_dict = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        return loader_dict
        

    def _balance_negative(self, pos_df, neg_df):
        # instantiate the balancer then set the ref and target
        balancer = DataBalancer()
        balancer.set_reference(pos_df)
        balancer.set_target(neg_df)

        # select features
        feature_list = ['tissueden', 'ViewPosition', 'Race']
        balancer.set_feature_list(feature_list)

        # set a 1:1 balance
        balancer.set_balance(1, 1)
        balancer.get_unique_levels()
        balancer.measure_slices()
        balancer.sample_slices()
        balanced_neg_df = balancer.merge_slices()
        return balanced_neg_df
        
    def objective_wrapper(self, **kwargs):
        """
        wrapper to pass the keyword arguments from the bayes opt package
        to the objective function as a dict since we're passing the params 
        to the model_function (not the objective function) as a dict.
        """
        # here kwargs are the direct output of the bayesian optimization func
        # so in this case, something like {learning_rate=0.01, dropout=0.5}
        # pop the first seed from the list
        bootstrap_seeds = np.random.randint(
            low=0, high=9999999, 
            size=self.n_bootstraps
        )

        self.run_num += 1

        print(f'running {self.n_bootstraps} bootstraps...')
        print(f'with seeds: {bootstrap_seeds}')

        run_evals = []

        for i, seed in enumerate(bootstrap_seeds):
            self.boot_num = i

            loader_dict = self._get_bootstrap_dataloader(seed)

            eval_metric = self.bootstrap_objective(loader_dict, kwargs)
            print(f'bootstrap {i}: {eval_metric:.4f}')
            run_evals.append(eval_metric)

        return np.mean(run_evals)
    
    def bootstrap_objective(self, loader_dict, kwargs): # i need to think of a name for this
        history = self.objective_function(
            model_type=self.model_type, 
            device=self.device, 
            loader_dict=loader_dict, 
            n_epochs=self.epochs_per_run, 
            save_dir=f"temp_{self.model_name}/", 
            monitor_metric=self.val_monitor_metric,
            seed=self.seed,
            kwargs=kwargs,
            last_phase="val" # tune on the validation set so we don't overfit to test set
        )
        # it's still called .test tho... so still load it the same way so i don't have to change all the code
        eval_metric = history[-1].test[self.val_monitor_metric]
        
        # if using loss, multiply by -1 so we can maximize it
        if self.val_monitor_metric.split('_')[-1] == "loss":
            eval_metric *= -1
        
        # log hyperparameter combination and eval metric (and if it's the final run)
        log_data(kwargs, history[-1].test)
        return eval_metric
    
    def get_demographic_dict(self, df):
        out_dict = dict()

        for feature in ['tissueden', 'ViewPosition', 'Race']:
            out_dict[feature] = dict(df[feature].value_counts())

        return out_dict
    
    def log_demographics(self, df_dict):
        out_dict = dict()

        for label_name, label_dict in df_dict.items():
            out_label_dict = dict()

            for set_name, set_df in label_dict.items():
                out_label_dict[set_name] = self.get_demographic_dict(set_df)

            out_dict[label_name] = out_label_dict

        # write out_dict to a json file
        dirname = f"./logs/{model_name}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        filename = f"demo_run{self.run_num}_boot{self.boot_num}.json"
        with open(os.path.join(dirname, filename), 'w') as f:
            json.dump(out_dict, f)

def log_data(param_dict, metric_dict, final: bool = False):
    # build the output dict
    log_dict = {"final": final, "params": param_dict}
    # log_dict.update(param_dict)
    log_dict.update(metric_dict)
    
    # convert the dict to a json string
    json_str = json.dumps(log_dict)
    
    # log the json string
    logging.info(json_str)
    print("Results logged...")