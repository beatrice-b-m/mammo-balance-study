import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
import json
import logging
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
        self.model_func = None
        self.objective_function = None

        self.df_pool_dict = None
        self.mask_factor = None
        self.batch_size = None
        self.transform = None

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
        
        
    def load_data(self, df_pool_dict: dict, mask_factor: str, batch_size: int, augment_dict: dict):
        self.df_pool_dict = df_pool_dict
        self.mask_factor = mask_factor
        self.batch_size = batch_size
        self.transform = get_augment_list(augment_dict)
        print("Data loaded...")
        
    def set_model(self, device, model_func):
        """
        device, model, loader_dict, metric_collection, criterion,
        optimizer, n_epochs: int, save_dir, str or None = None,
        monitor_metric: str = "val_loss"
        """
        
        self.device = device
        self.model_func = model_func
                
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

    def _get_bootstrap_dataloader(self, seed, img_col: str = 'png_path'):
        pos_train = self.df_pool_dict['pos']['train'].sample(self.train_n, seed)
        pos_val = self.df_pool_dict['pos']['val'].sample(self.val_n, seed)
        pos_test = self.df_pool_dict['pos']['test'].sample(self.test_n, seed)

        if self.balance:
            neg_train = self._balance_negative(pos_train, self.df_pool_dict['neg']['train'])
            neg_val = self._balance_negative(pos_val, self.df_pool_dict['neg']['val'])

        else:
            neg_train = self.df_pool_dict['neg']['train'].sample(self.train_n, seed)
            neg_val = self.df_pool_dict['neg']['val'].sample(self.val_n, seed)

        # use the full neg test set we provide HANDLE THIS BEFOREHAND
        neg_test = self.df_pool_dict['neg']['test']

        # NEED TO PARSE THE IMG_COL AT SOME POINT SO IT"S ACTUALLY CORRECT!!!!!!
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
        feature_list = ['tissueden', 'ViewPosition', 'Ethnicity', 'Race']
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

        print(f'running {self.n_bootstraps} bootstraps...')
        print(f'with seeds: {bootstrap_seeds}')

        run_evals = []

        for i, seed in enumerate(bootstrap_seeds):

            loader_dict = self.get_bootstrap_loader(seed)

            eval_metric = self._temp_obj_wrapper(loader_dict, kwargs)
            print(f'bootstrap {i}: {eval_metric:.4f}')
            run_evals.append(eval_metric)

        return np.mean(run_evals)
    
    def _temp_obj_wrapper(self, loader_dict, kwargs): # i need to think of a name for this
        history = self.objective_function(
            model_func=self.model_func, 
            device=self.device, 
            loader_dict=loader_dict, 
            n_epochs=self.epochs_per_run, 
            save_dir=f"temp_{self.model_name}/", 
            monitor_metric=self.val_monitor_metric,
            seed=self.seed,
            model_pdict=kwargs,
            last_phase="val" # tune on the validation set so we don't overfit to test set
        )
        # it's still called .test tho... so still load it the same way so i don't have to change all the code
        eval_metric = history[-1].test[self.test_monitor_metric]
        
        # if using loss, multiply by -1 so we can maximize it
        if self.test_monitor_metric.split('_')[-1] == "loss":
            eval_metric *= -1
        
        # log hyperparameter combination and eval metric (and if it's the final run)
        log_data(kwargs, history[-1].test)
        return eval_metric

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