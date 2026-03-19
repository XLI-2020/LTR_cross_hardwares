import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score
import torch
import matplotlib.pyplot as plt




def calculate_linear_raw_scores(scores):
    print('runtimes: ', scores)
    scores_sorted = list(np.sort(scores)[::-1])

    # print('scores_sorted: ', scores_sorted)
    score_sorted_indices = []
    for idx, i in enumerate(scores):
        index = scores_sorted.index(i)
        score_sorted_indices.append(index)
        scores_sorted[index] = "abcde"

    print('scores: ', score_sorted_indices)

    return score_sorted_indices

tpch_Cost_runtime_comb_path = "./labeled_plans_tpch_part_COST_used.csv"

# imdb_Cost_runtime_comb_path = "./labeled_plans_imdb_part_COST_used.csv"

df = pd.read_csv(tpch_Cost_runtime_comb_path, index_col=0)

df_used = df[(~df['Cost'].isna())&(~df['Time'].isna())]

Job_nr_list = df_used["Job_nr"].unique().tolist()

ndcg_sk_list = []

topk_list = [3, 6, 10, 15]
mean_ndcg_list = []

median_ndcg_list = []

for topk in topk_list:
    for job_nr in Job_nr_list:

        job_df = df_used[df_used["Job_nr"] == job_nr]

        cost_labels = job_df['Cost'].values

        runtime_labels = job_df['Time'].values

        cost_labels = calculate_linear_raw_scores(cost_labels)
        print('####')

        runtime_labels = calculate_linear_raw_scores(runtime_labels)

        print('job_nr', job_nr)

        print('cost_labels: ', cost_labels)
        print('runt_labels: ', runtime_labels)

        ndcg_sk = ndcg_score(np.array([runtime_labels]), np.array([cost_labels]), k=topk)
        print(f'{job_nr}, ndcg_sk', ndcg_sk)

        # ndcg_tor = ndcg_wrap(cost_labels, runtime_labels)
        # print(f'{job_nr}, ndcg_tor', ndcg_tor)

        ndcg_sk_list.append(ndcg_sk)

    print("the number of jobs: ", len(Job_nr_list))
    print("mean ndsg over all queries: ", np.mean(ndcg_sk_list))

    ndcg_df = pd.DataFrame(ndcg_sk_list)

    ndcg_df.columns = ['NDCG']

    print(ndcg_df.describe())

    mean_ndcg = ndcg_df.describe().loc["mean", 'NDCG']
    mean_ndcg_list.append(mean_ndcg)

    median_ndcg = ndcg_df.describe().loc["50%", 'NDCG']
    median_ndcg_list.append(median_ndcg)

print('mean', mean_ndcg_list)
print('median', median_ndcg_list)

def plot_ndcg(topk_list, mean_ndcg_list, median_ndcg_list, db="tpch"):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    barWidth = 0.25

    br1 = np.arange(len(topk_list))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    ax.bar(br1, mean_ndcg, color='r', width=barWidth, edgecolor='k', label='Mean')

    ax.bar(br2, median_ndcg, color='g', width=barWidth, edgecolor='k', label='Median')

    # ax.bar(br3, df3["Time"]/1000, color ='b', width = barWidth, edgecolor ='k', label ='XL')

    # plt.bar(br3, CSE, color='b', width=barWidth,
    #         edgecolor='grey', label='CSE')
    plt.xlabel(f'K values ({db})', fontsize=20)
    plt.ylabel('NDCG@K', fontsize=20)
    plt.ylim(-0.01, 1.1)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
    plt.xticks([r + 0.5*barWidth for r in range(len(topk_list))], [topk_list[r] for r in range(len(topk_list))], fontsize=20)

    plt.legend(ncol=2, fontsize=10)
    plt.tight_layout()

    plt.show()
plot_ndcg(topk_list, mean_ndcg_list, median_ndcg_list, db="JOB")












