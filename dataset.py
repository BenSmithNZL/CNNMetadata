import config
import dropbox
import numpy as np
import os
import pandas as pd
import PIL
from sklearn import preprocessing
import torch
import urllib.request


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_frame, TRANSFORM, experiment):

        self.data_frame = data_frame
        self.TRANSFORM = TRANSFORM

        if experiment == 0:
            pass
        elif experiment == 1:
            pat = 'normalised'
        elif experiment == 2:
            pat = 'level_1_name'
        elif experiment == 3:
            pat = 'level_2_name'
        elif experiment == 4:
            pat = 'grid'
        else:
            raise Exception("Invaild experiment")

        self.experiment = experiment

        if experiment != 0:
            self.metadata_columns = self.data_frame.columns[self.data_frame.columns.str.contains(pat)]

    def __len__(self):

        return(len(self.data_frame))

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        image_file_name = (config.FILE_PATH + 'data/images/' + self.data_frame['file_name'][index])
        image = PIL.Image.open(image_file_name)
        image = self.TRANSFORM(image)
        label = self.data_frame['label'][index]

        if self.experiment == 0:
            return(image, label)
        else:
            metadata = torch.FloatTensor(self.data_frame.iloc[[index]][list(self.metadata_columns)].values.tolist()[0])
            return(image, label, metadata)


class DatasetPreparation:

    def __init__(self, TOKEN, mode=0, num_of_downloads=None):

        self.existing_filenames = os.listdir(config.FILE_PATH + 'data/images/')

        if mode == 0:
            print(f"Mode {mode} selected, using metadata_test.csv with no downloads")

            self.metadata_df = pd.read_csv(
                config.FILE_PATH + 'data/metadata/metadata_test.csv',
                delimiter=",")

        elif mode == 1:
            print(f"Mode {mode} selected, using existing files with some downloads")

            base = pd.read_csv(
                config.FILE_PATH + 'data/metadata/metadata_base.csv',
                delimiter=",")

            ex = base[base['file_name'].apply(lambda x: x in self.existing_filenames)]

            self.metadata_df = pd.concat([
                ex,
                base[base['source'] == "N"].sample(num_of_downloads),
                base[base['source'] == "P"].sample(num_of_downloads)])

        else:
            print(f"Mode {mode} selected, doing full download")

            self.metadata_df = pd.read_csv(
                config.FILE_PATH + 'data/metadata/metadata_base.csv',
                delimiter=",")

        self.metadata_df.to_csv(config.FILE_PATH + 'data/metadata/metadata_raw.csv', index=False)
        self.dbx = dropbox.Dropbox(TOKEN)

    def inaturalist_download(self, url, file_name):
        try:
            urllib.request.urlretrieve(url, config.FILE_PATH + 'data/images/' + file_name)
            image = PIL.Image.open(config.FILE_PATH + 'data/images/' + file_name)
            image.thumbnail((config.IMAGE_SIZE, config.IMAGE_SIZE))
            image.save(config.FILE_PATH + 'data/images/' + file_name)
            print(f"Downloaded {file_name}")
        except:
            print(f"Error downloading {file_name}")
            if os.path.exists(config.FILE_PATH + 'data/images/' + file_name):
                os.remove(config.FILE_PATH + 'data/images/' + file_name)

    def dropbox_download(self, file_name):
        try:
            with open(config.FILE_PATH + 'data/images/' + file_name, 'wb') as f:
                _, result = self.dbx.files_download(path='/NZ plant photos/' + file_name)
                f.write(result.content)
                image = PIL.Image.open(config.FILE_PATH + 'data/images/' + file_name)
                image.thumbnail((config.IMAGE_SIZE, config.IMAGE_SIZE))
                image.save(config.FILE_PATH + 'data/images/' + file_name)
                print(f"Downloaded {file_name}")
        except:
            print(f"Error downloading {file_name}")
            if os.path.exists(config.FILE_PATH + 'data/images/' + file_name):
                os.remove(config.FILE_PATH + 'data/images/' + file_name)

    def get_files(self):

        def normalise_data(data):
            return((data - np.min(data)) / (np.max(data) - np.min(data)))

        def grid_maker(normalised_latitude, normalised_longitude, normalised_elevation):
            grid = (str(int(normalised_latitude * 10)),
                    str(int(normalised_longitude * 10)),
                    str(int(normalised_elevation * 10)))
            return("_".join(grid))

        print(f"{len(self.existing_filenames)} images existing")
        download_list = self.metadata_df[self.metadata_df['file_name'].apply(lambda x: x not in self.existing_filenames)]
        print(f"Have {len(download_list)} images to download")

        print("##### \nDownloading from Dropbox")
        for row in download_list[download_list['source'] == "P"].itertuples():
            self.dropbox_download(row.file_name)

        print("##### \nDownloading from iNaturalist")
        for row in download_list[download_list['source'] == "N"].itertuples():
            self.inaturalist_download(row.url, row.file_name)

        print("##### \nDownloading complete")

        downloaded_filenames = os.listdir(config.FILE_PATH + 'data/images/')
        self.metadata_df = self.metadata_df[self.metadata_df['file_name'].apply(lambda x: x in downloaded_filenames)]

        self.metadata_df['normalised_latitude'] = normalise_data(self.metadata_df['decimal_latitude'])
        self.metadata_df['normalised_longitude'] = normalise_data(self.metadata_df['decimal_longitude'])
        self.metadata_df['normalised_elevation'] = normalise_data(self.metadata_df['elevation'])
        self.metadata_df = pd.get_dummies(self.metadata_df, columns=['level_1_name'])
        self.metadata_df = pd.get_dummies(self.metadata_df, columns=['level_2_name'])

        self.metadata_df['grid'] = self.metadata_df.apply(lambda row: grid_maker(
            row['normalised_latitude'],
            row['normalised_longitude'],
            row['normalised_elevation']), axis=1)

        self.metadata_df['grid'] = self.metadata_df['grid'].apply(lambda x: x.replace("_10_", "_9_"))
        self.metadata_df['grid'] = self.metadata_df['grid'].apply(lambda x: x.replace("10_", "9_"))
        self.metadata_df['grid'] = self.metadata_df['grid'].apply(lambda x: x.replace("_10", "_9"))
        self.metadata_df = pd.get_dummies(self.metadata_df, columns=['grid'])

        label_encoder = preprocessing.LabelEncoder()
        self.metadata_df['label'] = label_encoder.fit_transform(self.metadata_df['scientific_name'])

        self.metadata_df.to_csv(config.FILE_PATH + 'data/metadata/metadata_clean.csv', index=False)

