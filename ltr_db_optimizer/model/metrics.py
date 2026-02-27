import torch
import numpy as np



PADDED_Y_VALUE = -1

def ndcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE,
         filler_value=1.0):
    """
    Normalized Discounted Cumulative Gain at k.

    Compute NDCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for NDCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param filler_value: a filler NDCG value to use when there are no relevant items in listing
    :return: NDCG values for each slate and rank passed, shape [batch_size, len(ats)]
    """
    idcg = dcg(y_true, y_true, ats, gain_function, padding_indicator)
    ndcg_ = dcg(y_pred, y_true, ats, gain_function, padding_indicator) / idcg
    idcg_mask = idcg == 0
    ndcg_[idcg_mask] = filler_value  # if idcg == 0 , set ndcg to filler_value

    assert (ndcg_ < 0.0).sum() >= 0, "every ndcg should be non-negative"

    return ndcg_


def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=PADDED_Y_VALUE):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)


def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=PADDED_Y_VALUE):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg


def mrr(y_pred, y_true, ats=None, padding_indicator=PADDED_Y_VALUE):
    """
    Mean Reciprocal Rank at k.

    Compute MRR at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for MRR evaluation, if None, maximum rank is used
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: MRR values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    if ats is None:
        ats = [y_true.shape[1]]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    values, indices = torch.max(true_sorted_by_preds, dim=1)
    indices = indices.type_as(values).unsqueeze(dim=0).t().expand(len(y_true), len(ats))

    ats_rep = torch.tensor(data=ats, device=indices.device, dtype=torch.float32).expand(len(y_true), len(ats))

    within_at_mask = (indices < ats_rep).type(torch.float32)

    result = torch.tensor(1.0) / (indices + torch.tensor(1.0))

    zero_sum_mask = torch.sum(values) == 0.0
    result[zero_sum_mask] = 0.0

    result = result * within_at_mask

    return result


class LTRMetric:
    
    name: str = None
    
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        pass


class AverageBest(LTRMetric):
    
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        predicted_best = ground_truth[torch.arange(predicted_y.shape[0]), torch.argmax(predicted_y, dim = 1)]
        return torch.mean(predicted_best.type(torch.float16))
    
class HitsAtK(LTRMetric):
    def __init__(self, k=10):
        self.k = 10
        
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        
        def func(array):
            split_at = int(len(array)/2)
            return len(np.intersect1d(array[:split_at], array[split_at:]))
        k = self.k if predicted_y.shape[-1] > self.k else predicted_y.shape[-1]
        # get the indices of the best k predicted and true values, row-wise
        top_k_pred = torch.topk(predicted_y, k, dim=-1).indices
        top_k_true = torch.topk(ground_truth, k, dim=-1).indices
        # calculate the number of same elements (row-wise), sum it up and divide it by the number of regarded elements
        temp_all = torch.cat((top_k_pred,top_k_true), 1).detach().numpy()
        return np.sum(np.apply_along_axis(func, 1, temp_all))/torch.numel(top_k_true)
    
class FoundBest(LTRMetric):
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        
        return torch.sum(torch.argmax(predicted_y, dim=1) == torch.argmax(ground_truth, dim=1))
    
class NDCG(LTRMetric):
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        pass
    
class PositionK(LTRMetric):
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        def func(array):
            print(111, np.where(array[:-1] == array[-1]))
            return np.where(array[:-1] == array[-1])[0]+1
        pred_sort = torch.argsort(predicted_y, descending = True)
        print('predicted_y, pred_sort', predicted_y, pred_sort)
        true_sort = torch.argsort(ground_truth, descending = True)[:,0].reshape(-1,1)
        print('ground_truth, true_sort', ground_truth, true_sort)
        temp_all = torch.cat((pred_sort,true_sort), 1).detach().numpy()
        print('temp_all', temp_all)
        res = np.max(np.apply_along_axis(func, 1, temp_all))
        print('res', res)
        return res
        
class FoundBestK(LTRMetric):
    def __init__(self, k=5):
        self.k = k
    
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        k = self.k if predicted_y.shape[-1] >= self.k else predicted_y.shape[-1] 
        temp_pred = torch.argsort(predicted_y, dim=1, descending=True).detach().numpy()
        temp_true = torch.argsort(ground_truth, dim=1, descending=True).detach().numpy()[:,:k]
        temp_all = np.concatenate((temp_pred,temp_true), 1)
        for row in temp_all:
            for idx, number in enumerate(row[:-k]):
                if number in row[-k:]:
                    return idx+1
    
class TopKFound(LTRMetric):
    def __init__(self, k=5):
        self.k = k
        
    def calculate(self, predicted_y: torch.Tensor, ground_truth: torch.Tensor) -> float:
        assert predicted_y.shape == ground_truth.shape
        k = self.k if predicted_y.shape[-1] >= self.k else predicted_y.shape[-1] 
        
        def func(array):
            print('np.where(array[1:] == array[0])', np.where(array[1:] == array[0]))
            return len(np.where(array[1:] == array[0])[0])
        
        temp_pred = torch.argmax(predicted_y, dim=1).detach().numpy().reshape(-1,1)
        print('temp_pred', temp_pred)
        temp_true = torch.argsort(ground_truth, dim=1, descending=True).detach().numpy()[:,:k]
        print('temp_true', temp_true)
        temp_all = np.concatenate((temp_pred,temp_true), 1)
        print('temp_all', temp_all)
        return np.sum(np.apply_along_axis(func, 1, temp_all))



# pk = TopKFound()
#
# true = torch.tensor([[10, 9,8,7,6,5,4]])
# # pred = torch.tensor([[0.2, 0.5]])
#
# pred = torch.tensor([[0.9,0.1, 0.2, 0.5, 0.55, 0.6, 0.7, ]])
#
# # pred = torch.tensor([[0.1,  0.5, 0.2, 0.9]])
#
#
#
# pk_value = pk.calculate(pred, true)
#
# print('pk value: ', pk_value)
    
    