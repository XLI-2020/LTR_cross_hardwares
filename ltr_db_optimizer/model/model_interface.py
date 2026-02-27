import math
import torch
import itertools
import numpy as np
import random
import os
import pickle
from ltr_db_optimizer.ext.ptranking.ltr_adhoc.listwise.listnet import ListNet
from ltr_db_optimizer.model.featurizer_dict import table_info
from ltr_db_optimizer.model.model_structures.comparison_net2 import LTRComparisonNet ##changed

from sklearn.metrics import ndcg_score

from ltr_db_optimizer.ext.ptranking.data.data_utils import LABEL_TYPE

from datetime import datetime
from  ltr_db_optimizer.extra_utils import get_the_split_of_jobs_list, ndcg_wrap



def create_model(LossFunction, **kwargs):
    class ModelInterface(LossFunction):

        def __init__(self, epochs=201, sample_size=20, batch_size=64, workload=None, name="", folder="", **kwargs):
            super().__init__(sf_para_dict={"sf_id": 'pointsf', "opt": None, "lr": None}, **kwargs)
            self.epochs = epochs
            self.sample_size = sample_size
            self.batch_size = batch_size
            self.optimizer = None
            self.device = None

            self.workload = workload

            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

            print('CUDA available:', torch.cuda.is_available())
            self.name = name
            self.folder = folder


        def test_loss_function(self, pred, true):
            pred_results = pred
            curr_y = true
            loss = self.custom_loss_function(pred_results, curr_y, train=False, label_type=LABEL_TYPE.MultiLabel, presort=False)
            print('test loss: ', loss.item())


        def fit(self, X_train_vecs, X_train_tree, y_train, X_valid_vecs, X_valid_trees, y_valid,  use_presort=False,
                use_scheduler=False, optimizer="adam"):
            # X_test_vecs, X_test_trees, y_test,
            input_dim_1 = 10  # 9#10
            input_dim_2 = 6  # 4 #3# changed: input_dim_2 = 4
            model_archi_name = self.name.split("MODEL_")[1].split("_")[0]
            print('model_archi_name: ', model_archi_name)

            if model_archi_name == "HM":
                self.net = LTRComparisonNet(input_dim_1, input_dim_2).to(self.device)
                print('load LTRComparisonNet HM!!!')
                # self.net_ndcg = LTRComparisonNet(input_dim_1, input_dim_2).to(self.device)
            else:
                self.net = eval(model_archi_name)(input_dim_1, input_dim_2).to(self.device)
                print(f'load XL proposed model:{model_archi_name}!!!')
                print('load LTRComparisonNet XL!!!')

            if optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.net.parameters())
            elif optimizer == "adagrad":
                self.optimizer = torch.optim.Adagrad(self.net.parameters())
            elif optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)
            elif optimizer == "rmsprop":
                self.optimizer = torch.optim.RMSprop(self.net.parameters())

            if use_scheduler:
                scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

            if not os.path.exists(self.folder + "/" + self.name):
                os.makedirs(self.folder + "/" + self.name)

            best_ndcg = 0
            min_valid_loss = 1e8
            best_ndcg_epoch = None
            best_validloss_epoch = None

            result_dict = {}

            print('start training: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            all_training_loss = []
            all_training_ndcg = []
            all_training_ndcg_new = []

            all_valid_loss = []
            all_valid_ndcg = []
            all_valid_ndcg_new = []

            for epoch in range(self.epochs):
                print('epoch: ', epoch)
                train_loss_epoch = []
                train_ndcg_epoch = []
                train_ndcg_new_epoch = []

                num_of_train_samples =[]

                self.net.train()

                total_curr_job_numbers = get_the_split_of_jobs_list(X_train_vecs, train_ratio=0.5, workload=self.workload)

                print('curr train jobs and length: ',  len(total_curr_job_numbers), total_curr_job_numbers[:10])

                num_batches = len(total_curr_job_numbers) // self.batch_size if len(total_curr_job_numbers) % self.batch_size == 0 else (len(total_curr_job_numbers) // self.batch_size + 1)
                print('number of batches: ', num_batches)

                for batch_i in range(num_batches):
                    print('batch_i: ', batch_i)
                    curr_job_numbers = total_curr_job_numbers[batch_i * self.batch_size:(batch_i + 1) * self.batch_size]
                    print('curr_job_numbers: ', curr_job_numbers[:5])

                    sample_size_list = [6, 20]
                # for curr_sample_size in sample_size_list:
                    curr_sample_size = self.sample_size

                    batch_x_p = []
                    batch_x_q = []
                    batch_y = []
                    for idx, x in enumerate(curr_job_numbers):
                        curr_keys = list(X_train_vecs[x].keys())
                        print('len of curr_keys: ', len(curr_keys))

                        if len(curr_keys) < curr_sample_size:
                            continue
                        if len(curr_keys) > curr_sample_size:
                            curr_keys = random.sample(curr_keys, curr_sample_size)

                        curr_x = [X_train_tree[x][key] for key in curr_keys]
                        if len(curr_x) == 1:
                            continue

                        print('the first 5 training query vectors: ', [np.array(X_train_vecs[x][key]) for key in curr_keys][:5])
                        curr_x_vec = [X_train_vecs[x][key] for key in curr_keys]

                        batch_x_p = batch_x_p + curr_x
                        batch_x_q = batch_x_q + curr_x_vec

                        curr_y = [float(y_train[key]) for key in curr_keys]

                        batch_y.append(curr_y)

                    batch_x_q = torch.Tensor(batch_x_q).to(self.device)
                    print('batch_x_q shp: ', batch_x_q.shape)
                    print('batch_x_p: ', len(batch_x_p))
                    batch_y = torch.Tensor(batch_y).to(self.device)
                    print('batch_y shp: ', batch_y.shape)
                    if len(batch_x_p) == 0:
                        continue

                    pred_results = self.net(batch_x_q, batch_x_p, plan_num=curr_sample_size)
                    pred_results = torch.squeeze(pred_results, -1)

                    if use_presort:
                        curr_y, batch_ideal_desc_inds = torch.sort(batch_y, dim=1, descending=True)
                        pred_results = torch.gather(pred_results, dim=1, index=batch_ideal_desc_inds)

                    print('y predict: ', pred_results.shape, pred_results[:5])
                    print('y true: ', batch_y.shape, batch_y[:5])
                    print('######')

                    loss = self.custom_loss_function(pred_results, batch_y, train=True, label_type=LABEL_TYPE.MultiLabel, presort=use_presort)
                    print('loss: ', round(loss.item()/batch_y.shape[0], 2))
                    train_loss_epoch.append(loss.item())
                    num_of_train_samples.append(batch_y.shape[0])

                    # print('y_predict, y_true for ndcg calculation: ',pred_results[0].detach().cpu().numpy(), curr_y[0].detach().cpu().numpy() )
                    train_ndcg_value = ndcg_wrap(pred_results.detach().cpu(), batch_y.detach().cpu())
                    print('train ndcg: ', train_ndcg_value.mean().item())
                    train_ndcg_epoch.append(train_ndcg_value.sum().item())

                    train_ndcg_value_new = ndcg_score(batch_y.detach().cpu().numpy(), pred_results.detach().cpu().numpy())
                    print('train ndcg new: ', train_ndcg_value_new)
                    train_ndcg_new_epoch.append(train_ndcg_value_new*batch_y.shape[0])
                    print('######')


                avg_train_loss_epoch = round(np.sum(train_loss_epoch)/np.sum(num_of_train_samples), 2)
                print(f'Epoch {epoch}: average training loss:', avg_train_loss_epoch)
                all_training_loss.append((epoch, avg_train_loss_epoch))
                avg_train_ndcg_epoch = round(np.sum(train_ndcg_epoch)/np.sum(num_of_train_samples), 2)
                print(f'Epoch {epoch}: average training ndcg:', avg_train_ndcg_epoch)
                all_training_ndcg.append((epoch, avg_train_ndcg_epoch))

                avg_train_ndcg_new_epoch = round(np.sum(train_ndcg_new_epoch)/np.sum(num_of_train_samples), 2)
                print(f'Epoch {epoch}: average training ndcg new:', avg_train_ndcg_new_epoch)
                all_training_ndcg_new.append((epoch, avg_train_ndcg_new_epoch))


                ### next go to validation of the model
                valid_ndcg_epoch = []
                valid_loss_epoch = []
                valid_ndcg_new_epoch = []
                num_of_actual_valid_samples_epoch = []

                with torch.no_grad():

                    self.net.eval()

                    ignored = 0

                    print('current valid jobs and length: ', len(list(X_valid_vecs.keys())), list(X_valid_vecs.keys())[:10])


                    valid_job_numbers = list(X_valid_vecs.keys())

                    num_valid_batches = len(valid_job_numbers) // self.batch_size if len(valid_job_numbers) % self.batch_size == 0 else (len(valid_job_numbers) // self.batch_size + 1)

                    print('number of valid batches: ', num_valid_batches)

                    for valid_batch_i in range(num_valid_batches):

                        curr_valid_job_numbers = valid_job_numbers[valid_batch_i * self.batch_size:(valid_batch_i + 1) * self.batch_size]

                        valid_batch_x_p = []
                        valid_batch_x_q = []
                        valid_batch_y = []

                        for idx, x_valid in enumerate(curr_valid_job_numbers):
                            curr_plan_keys = list(X_valid_vecs[x_valid].keys())

                            if len(curr_plan_keys) < self.sample_size:
                                continue
                            if len(curr_plan_keys) > self.sample_size:
                                curr_plan_keys = random.sample(curr_plan_keys, self.sample_size)


                            valid_data_tree = [X_valid_trees[x_valid][key] for key in curr_plan_keys]

                            valid_data_vec = [X_valid_vecs[x_valid][key] for key in curr_plan_keys]

                            if not valid_data_vec or len(valid_data_tree) == 1:
                                ignored += 1
                                continue
                            # print('original type of valid_data_vec:', type(valid_data_vec))
                            # valid_data_vec = torch.Tensor(valid_data_vec).to(self.device)
                            # test_data_tree = torch.Tensor(test_data_tree).to(self.device)
                            y_true = [float(y_valid[key]) for key in curr_plan_keys]

                            valid_batch_x_p = valid_batch_x_p + valid_data_tree
                            valid_batch_x_q = valid_batch_x_q + valid_data_vec
                            valid_batch_y.append(y_true)

                        valid_batch_x_q = torch.Tensor(valid_batch_x_q).to(self.device)

                        valid_batch_y = torch.Tensor(valid_batch_y).to(self.device)

                        y_predicted = self.net.predict_all(valid_batch_x_q, valid_batch_x_p, plan_num=self.sample_size)


                        print('######')
                        print('validate y_predicted', y_predicted.shape, y_predicted[:5])
                        print('validate y_true: ', valid_batch_y.shape, valid_batch_y[:5])
                        valid_loss = self.custom_loss_function(y_predicted, valid_batch_y, train=False, label_type=LABEL_TYPE.MultiLabel)

                        print('validate loss: ', round(valid_loss.item()/valid_batch_y.shape[0], 2))
                        valid_loss_epoch.append(valid_loss.item())

                        num_of_actual_valid_samples_epoch.append(valid_batch_y.shape[0])
                        # print('validate y_predict, y_true for ndcg calculation: ', y_predicted[0].detach().cpu().numpy(), y_true[0].detach().cpu().numpy())

                        valid_ndcg = ndcg_wrap(y_predicted.detach().cpu(), valid_batch_y.detach().cpu())
                        print('validate ndcg', valid_ndcg.mean().item())
                        valid_ndcg_epoch.append(valid_ndcg.sum().item())

                        valid_ndcg_new = ndcg_score(valid_batch_y.detach().cpu().numpy(), y_predicted.detach().cpu().numpy())
                        print('validate ndcg new: ', valid_ndcg_new)
                        valid_ndcg_new_epoch.append(valid_ndcg_new*valid_batch_y.shape[0])
                        print('######')

                avg_valid_ndcg = round(np.sum(valid_ndcg_epoch)/np.sum(num_of_actual_valid_samples_epoch), 2)
                print(f'Epoch {epoch}: average valid ndcg: ', avg_valid_ndcg)


                avg_valid_loss = round(np.sum(valid_loss_epoch)/np.sum(num_of_actual_valid_samples_epoch), 2)
                print(f'Epoch {epoch}: average valid loss: ', avg_valid_loss)

                avg_valid_ndcg_new = round(np.sum(valid_ndcg_new_epoch)/np.sum(num_of_actual_valid_samples_epoch), 2)
                print(f'Epoch {epoch}: average valid ndcg new: ', avg_valid_ndcg_new)

                all_valid_loss.append((epoch, avg_valid_loss))
                all_valid_ndcg.append((epoch, avg_valid_ndcg))
                all_valid_ndcg_new.append((epoch, avg_valid_ndcg_new))


                if avg_valid_ndcg > best_ndcg:
                    print("save model because of ndcg!")
                    best_ndcg = avg_valid_ndcg
                    best_ndcg_epoch = epoch
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/best_avg_ndcg.pth")

                if avg_valid_loss < min_valid_loss:
                    print("save model because of valid loss!")
                    min_valid_loss = avg_valid_loss
                    best_validloss_epoch = epoch
                    torch.save(self.net.state_dict(), f"{self.folder}/{self.name}/min_avg_valid_loss.pth")

                if use_scheduler:
                    scheduler.step()
                    print('current lr:', scheduler.get_lr())


            result_dict['train_loss'] = all_training_loss
            result_dict['valid_loss'] = all_valid_loss
            result_dict['train_ndcg'] = all_training_ndcg
            result_dict['valid_ndcg'] = all_valid_ndcg

            result_dict['train_ndcg_new'] = all_training_ndcg_new
            result_dict['valid_ndcg_new'] = all_valid_ndcg_new

            result_dict['best_ndcg_epoch'] = best_ndcg_epoch
            result_dict['best_validloss_epoch'] = best_validloss_epoch
            with open(self.folder + "/" + self.name + "/train_and_valid_info.pickle", "wb") as f:
                pickle.dump(result_dict, f)

            print('finish training: ', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


        def test(self, X_test_vecs, X_test_trees, y_test):

            self.net.load_state_dict(torch.load(f"{self.folder}/{self.name}/min_avg_valid_loss.pth"))

            valid_ndcg_epoch = []
            valid_loss_epoch = []
            valid_ndcg_new_epoch = []
            num_of_actual_valid_samples_epoch = []

            with torch.no_grad():

                self.net.eval()

                ignored = 0

                print('total test jobs and length: ', len(list(X_test_vecs.keys())), list(X_test_vecs.keys())[:10])

                valid_job_numbers = list(X_test_vecs.keys())

                num_valid_batches = len(valid_job_numbers) // self.batch_size if len(valid_job_numbers) % self.batch_size == 0 else (len(valid_job_numbers) // self.batch_size + 1)

                print('number of test batches: ', num_valid_batches)

                for valid_batch_i in range(num_valid_batches):

                    curr_valid_job_numbers = valid_job_numbers[valid_batch_i * self.batch_size:(valid_batch_i + 1) * self.batch_size]

                    valid_batch_x_p = []
                    valid_batch_x_q = []
                    valid_batch_y = []

                    for idx, x_valid in enumerate(curr_valid_job_numbers):
                        curr_plan_keys = list(X_test_vecs[x_valid].keys())

                        if len(curr_plan_keys) < self.sample_size:
                            continue
                        if len(curr_plan_keys) > self.sample_size:
                            # curr_plan_keys = random.sample(curr_plan_keys, self.sample_size)
                            curr_plan_keys = curr_plan_keys[:self.sample_size]  # avoid randomness for comparison among methods

                        valid_data_tree = [X_test_trees[x_valid][key] for key in curr_plan_keys]

                        valid_data_vec = [X_test_vecs[x_valid][key] for key in curr_plan_keys]

                        if not valid_data_vec or len(valid_data_tree) == 1:
                            ignored += 1
                            continue
                        # print('original type of valid_data_vec:', type(valid_data_vec))
                        # valid_data_vec = torch.Tensor(valid_data_vec).to(self.device)
                        # test_data_tree = torch.Tensor(test_data_tree).to(self.device)
                        y_true = [float(y_test[key]) for key in curr_plan_keys]

                        valid_batch_x_p = valid_batch_x_p + valid_data_tree
                        valid_batch_x_q = valid_batch_x_q + valid_data_vec
                        valid_batch_y.append(y_true)

                    valid_batch_x_q = torch.Tensor(valid_batch_x_q).to(self.device)

                    valid_batch_y = torch.Tensor(valid_batch_y).to(self.device)

                    y_predicted = self.net.predict_all(valid_batch_x_q, valid_batch_x_p, plan_num=self.sample_size)

                    print('######')
                    print('test y_predicted', y_predicted.shape, y_predicted[:5])
                    print('test y_true: ', valid_batch_y.shape, valid_batch_y[:5])
                    valid_loss = self.custom_loss_function(y_predicted, valid_batch_y, train=False, label_type=LABEL_TYPE.MultiLabel)

                    print('test loss: ', round(valid_loss.item() / valid_batch_y.shape[0], 2))
                    valid_loss_epoch.append(valid_loss.item())

                    num_of_actual_valid_samples_epoch.append(valid_batch_y.shape[0])
                    # print('validate y_predict, y_true for ndcg calculation: ', y_predicted[0].detach().cpu().numpy(), y_true[0].detach().cpu().numpy())

                    valid_ndcg = ndcg_wrap(y_predicted.detach().cpu(), valid_batch_y.detach().cpu())
                    print('test ndcg', valid_ndcg.mean().item())
                    valid_ndcg_epoch.append(valid_ndcg.sum().item())

                    valid_ndcg_new = ndcg_score(valid_batch_y.detach().cpu().numpy(), y_predicted.detach().cpu().numpy())
                    print('test ndcg new: ', valid_ndcg_new)
                    valid_ndcg_new_epoch.append(valid_ndcg_new * valid_batch_y.shape[0])
                    print('######')

            avg_valid_ndcg = round(np.sum(valid_ndcg_epoch) / np.sum(num_of_actual_valid_samples_epoch), 2)
            print(f'Test: average valid ndcg: ', avg_valid_ndcg)

            avg_valid_ndcg_new = round(np.sum(valid_ndcg_new_epoch) / np.sum(num_of_actual_valid_samples_epoch), 2)
            print(f'Test: average valid ndcg sk: ', avg_valid_ndcg_new)

            avg_valid_loss = round(np.sum(valid_loss_epoch) / np.sum(num_of_actual_valid_samples_epoch), 2)
            print(f'Test: average valid loss: ', avg_valid_loss)

            test_result_dict = {}
            test_result_dict['avg_test_loss'] = avg_valid_loss

            test_result_dict['avg_test_ndcg'] = avg_valid_ndcg

            test_result_dict['avg_test_ndcg_new'] = avg_valid_ndcg_new

            with open(self.folder + "/" + self.name + "/test_info.pickle", "wb") as f:
                pickle.dump(test_result_dict, f)


            return avg_valid_loss, avg_valid_ndcg, avg_valid_ndcg_new


    return ModelInterface(**kwargs)
