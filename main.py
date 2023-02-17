""" Imports """
import config
from dataset import Dataset, DatasetPreparation
from datetime import datetime
from evaluate import evaluate_baseline, evaluate_metadata
import json
from models import BaselineModel, MetadataModel
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import statistics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import warnings

""" Retrive dataset and download images """
dataset_preparation = DatasetPreparation(TOKEN=config.TOKEN, mode=0, num_of_downloads=1)
dataset_preparation.get_files()
metadata_df = pd.read_csv(config.FILE_PATH + 'data/metadata/metadata_clean.csv', delimiter=",")

""" Get constants """
NUM_OF_LEVEL_1_NAMES = sum(metadata_df.columns.str.contains(pat='level_1_name'))
NUM_OF_LEVEL_2_NAMES = sum(metadata_df.columns.str.contains(pat='level_2_name'))
NUM_OF_GRID_NAMES = sum(metadata_df.columns.str.contains(pat='grid'))

label_encoder = preprocessing.LabelEncoder()
metadata_df['label'] = label_encoder.fit_transform(metadata_df['scientific_name'])
NUM_OF_CLASSES = len(metadata_df['label'].unique())
class_weights = len(metadata_df) / metadata_df['label'].value_counts().sort_index()

""" Create master """
master = {

    'model_name': ['baseline_model',
                   'model_1',
                   'model_2',
                   'model_3',
                   'model_4'],

    'model': [BaselineModel(NUM_OF_CLASSES),
              MetadataModel(NUM_OF_CLASSES, 3),
              MetadataModel(NUM_OF_CLASSES, NUM_OF_LEVEL_1_NAMES),
              MetadataModel(NUM_OF_CLASSES, NUM_OF_LEVEL_2_NAMES),
              MetadataModel(NUM_OF_CLASSES, NUM_OF_GRID_NAMES)],

    'dataset': [Dataset(metadata_df, config.TRANSFORM, 0),
                Dataset(metadata_df, config.TRANSFORM, 1),
                Dataset(metadata_df, config.TRANSFORM, 2),
                Dataset(metadata_df, config.TRANSFORM, 3),
                Dataset(metadata_df, config.TRANSFORM, 4)]
}

raw_results = {
    'model_name': [],
    'fold': [],
    'predicted': [],
    'true': []
}

test_metrics = {
    'model_name': [],
    'fold': [],
    'accuracy': [],
    'balanced_accuracy': [],
    'weighted_f1': [],
    'weighted_precision': [],
    'weighted_recall': []
}


""" K fold """
torch.manual_seed(config.SEED)

splits = KFold(
    n_splits=config.K,
    shuffle=True,
    random_state=config.SEED)

start_time = datetime.now()
print("\n###### Cross-validation started at: ", start_time.strftime("%d/%m/%Y %H:%M:%S"), " ######")

for fold, (train_index, test_index) in enumerate(splits.split(np.arange(len(metadata_df)))):

    if fold >= (config.STARTING_FOLD - 1):

        print(f'\n###### Starting fold {fold + 1} of {config.K} ######')

        for index, model_name in enumerate(master['model_name']):

            print(f'### Evaluating {model_name}')

            if model_name == "baseline_model":
                predicted, true = evaluate_baseline(
                    master['dataset'][index],
                    train_index,
                    test_index,
                    master['model'][index],
                    class_weights)

            else:
                predicted, true = evaluate_metadata(
                    master['dataset'][index],
                    train_index,
                    test_index,
                    master['model'][index],
                    class_weights)

            raw_results['model_name'].append(model_name)
            raw_results['fold'].append(fold + 1)
            raw_results['predicted'].append(predicted)
            raw_results['true'].append(true)

            test_metrics['model_name'].append(model_name)
            test_metrics['fold'].append(fold + 1)
            test_metrics['accuracy'].append(accuracy_score(true, predicted))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_metrics['balanced_accuracy'].append(balanced_accuracy_score(true, predicted))
                test_metrics['weighted_f1'].append(f1_score(true, predicted, average='weighted', zero_division=0))
                test_metrics['weighted_precision'].append(precision_score(true, predicted, average='weighted', zero_division=0))
                test_metrics['weighted_recall'].append(recall_score(true, predicted, average='weighted', zero_division=0))


end_time = datetime.now()
print("\n###### Cross-validation finished at: ", end_time.strftime("%d/%m/%Y %H:%M:%S"), " ######")

with open(config.FILE_PATH + "results/raw_results/raw_results.json", "w") as outfile:
    json.dump(raw_results, outfile)

""" Summarised results """
test_metrics = pd.DataFrame(test_metrics)
summarised_metrics = pd.DataFrame()

for model_name in master['model_name']:
    model_test_metrics = test_metrics[test_metrics['model_name'] == model_name]

    model_summerised_metrics = pd.DataFrame(
        [[model_name, 'accuracy', np.mean(model_test_metrics['accuracy']), statistics.stdev(model_test_metrics['accuracy'])],
         [model_name, 'balanced_accuracy', np.mean(model_test_metrics['balanced_accuracy']), statistics.stdev(model_test_metrics['balanced_accuracy'])],
         [model_name, 'weighted_f1', np.mean(model_test_metrics['weighted_f1']), statistics.stdev(model_test_metrics['weighted_f1'])],
         [model_name, 'weighted_precision', np.mean(model_test_metrics['weighted_precision']), statistics.stdev(model_test_metrics['weighted_precision'])],
         [model_name, 'weighted_recall', np.mean(model_test_metrics['weighted_recall']), statistics.stdev(model_test_metrics['weighted_recall'])]])

    summarised_metrics = pd.concat([summarised_metrics, model_summerised_metrics])

summarised_metrics.columns = ['model_name', 'metric', 'mean', 'standard_deviations']


""" Confusion matricies """
raw_results = pd.DataFrame(raw_results)


def flatten(l):
    return [item for sublist in l for item in sublist]


for model_name in master['model_name']:

    cm = confusion_matrix(flatten(raw_results[raw_results['model_name'] == model_name]['true']),
                          flatten(raw_results[raw_results['model_name'] == model_name]['predicted']))
    cm = pd.DataFrame(cm)
    cm.columns = label_encoder.classes_
    cm.index = label_encoder.classes_
    cm.to_csv(config.FILE_PATH + "results/confusion_matricies/" + model_name + "_confusion_matrix.csv", index=True)


""" Experiment info """
experiment_info = {
    'parameter': [
        'start_datetime',
        'end_datetime',
        'seconds_elapsed',
        'minutes_elapsed',
        'hours_elapsed',
        'days_elapsed',
        'num_of_obs',
        'starting_fold'],

    'value': [
        start_time.strftime("%d/%m/%Y %H:%M:%S"),
        end_time.strftime("%d/%m/%Y %H:%M:%S"),
        (end_time - start_time).total_seconds(),
        (end_time - start_time).total_seconds() / 60,
        (end_time - start_time).total_seconds() / 3600,
        (end_time - start_time).total_seconds() / 86400,
        len(metadata_df),
        config.STARTING_FOLD]
}


""" Save to disk """
raw_results.to_csv(config.FILE_PATH + "results/raw_results/raw_results.csv", index=False)
test_metrics.to_csv(config.FILE_PATH + "results/test_metrics.csv", index=False)
summarised_metrics.to_csv(config.FILE_PATH + "results/summarised_metrics.csv", index=False)
experiment_info = pd.DataFrame(experiment_info)
experiment_info.to_csv(config.FILE_PATH + "results/experiment_info.csv", index=False)
