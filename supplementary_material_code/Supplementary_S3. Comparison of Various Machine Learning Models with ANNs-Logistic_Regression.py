
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import multiprocessing
# from multiprocessing import Pool
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score


def expand_df(df, first, last):
    return [list(row[1:]) for row in df.loc[:, first:last].itertuples()]


def get_training_set(complete_set, train_index_list, first, last):
    train_set = list()
    for ind in train_index_list:
        for item in expand_df(complete_set[ind], first, last):
            train_set.append(item)
    return train_set


def get_adr_values(complete_set, train_index_list):
    adr = list()
    for ind in train_index_list:
        for rows in complete_set[ind][[complete_set[ind].columns[1]]].itertuples():
            adr.append(list(rows[1:])[0])

    return adr


def compute_accuracy(train_set, train_adr):
    #clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf = LogisticRegression()
    clf.fit(train_set, train_adr)
    return clf


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_start(adr, file2, validation_data, validation_data_adr, file):
    #sample_length = 11164

    first = file.columns[1]
    last = file.columns[-1]

    sample = file.loc[:, first:last]
    sample = sample.loc[:, first:last].reset_index(drop=True)

    print('Starting: ', adr)
    combined_sample = pd.concat([file2[['name', adr]].reset_index(drop=True), sample], axis=1)

    training_set = expand_df(combined_sample, first, last)
    training_adr = [row for row in combined_sample.loc[:, adr]]
    test_set = expand_df(validation_data, first, last)
    test_adr = [row for row in validation_data_adr.loc[:, adr]]

    print('Training Model: ', adr)

    model = compute_accuracy(training_set, training_adr)

    print('Training Model Done: ', adr)

    pred_adr = model.predict(test_set)
    accuracy = accuracy_score(pred_adr, test_adr)
    f1 = f1_score(test_adr, pred_adr, average='macro')
    precision = precision_score(test_adr, pred_adr, average='macro')
    roc_auc = roc_auc_score(test_adr, pred_adr)

    df = pd.DataFrame({
        'name': validation_data.loc[:, 'name'],
        'pred': pred_adr
    })

    df_acc = pd.DataFrame([[adr, accuracy, f1, precision, roc_auc]], columns=['ADR', 'Accuracy', 'F1', 'Precision', 'ROC'])

    print('Creating file:', adr)

    df.to_csv(adr + '.csv')
    df_acc.to_csv(adr + '_acc.csv')

    print('Done:', adr)


if __name__ == "__main__":
    base_path = '/sas/vidhya/susmitha/ADR_Prediction/final_results/Logistic Regression/pca/subset_1/' # 'C:\\Users\\Aditya R\\Downloads\\'
    pca_path = 'dataset_after_pca_1000_features.csv'  # 'principal_component_analysis_dataset.csv'
    adr_path = 'ADR_dataset_for_training_subset_1.csv'
    pred_subset_path = 'dataset_for_validation_after_pca_1000_features.csv'  # 'prediction_subset_two.csv'
    adr_subset_path = 'ADR_validation_for_validation_subset_1.csv'

    file = pd.read_csv(base_path + pca_path,index_col = 0)
    #file = file.rename(columns={file.columns[0]:'name'})
    file2 = pd.read_csv(base_path + adr_path,index_col = 0)
    file2 = file2.rename(columns={file2.columns[0]: 'name'})

    validation_data_pca = pd.read_csv(base_path + pred_subset_path, index_col = 0)
    validation_data_pca = validation_data_pca.rename(columns={validation_data_pca.columns[0]: 'name'})

    validation_data_adr = pd.read_csv(base_path + adr_subset_path,index_col = 0)
    validation_data_adr = validation_data_adr.rename(columns={validation_data_adr.columns[0]: 'name'})
    #pd.set_printoptions(max_columns=10)
    #sprint("training_
    print(file.head())
    print(file2.head())
    print(validation_data_pca.head())
    print(validation_data_adr.head())
    #print("training_adr")
    #print(validation_data_pca.head())
    #print("testing_dataset")
    #file2.head()
    #print("testing_adr")
    #validation_data_adr.head()
    

    adrs_all = [e for e in file2.columns[1:]]

    chunks = list(divide_chunks(adrs_all,15))

    #with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
    #    result = pool.map(process_start, )
    for adrs in chunks:
        jobs = []
        start_time = time.time()
        for adr in adrs:
            p = multiprocessing.Process(target=process_start, args=(adr, file2, validation_data_pca, validation_data_adr, file))
            jobs.append(p)
            p.start()
        print('Joining...')
        for job in jobs:
            job.join()
        print("--- %s seconds ---" % (time.time() - start_time))

