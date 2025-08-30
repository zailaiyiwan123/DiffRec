import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.rec_base_dataset_builder import RecBaseDatasetBuilder
# from minigpt4.datasets.datasets.laion_dataset import LaionDataset
# from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset

from minigpt4.datasets.datasets.rec_datasets import MovielensDataset, MovielensDataset_stage1, AmazonDataset, MoiveOOData, MoiveOOData_sasrec, AmazonOOData, AmazonOOData_sasrec, AmazonOOData_rating

# @registry.register_builder("movielens")
# class MovielensBuilder(RecBaseDatasetBuilder):
#     train_dataset_cls = MovielensDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/movielens/default.yaml",
#     }

#     def build_datasets(self):
#         # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
#         logging.info("Building datasets...")
#         self.build_processors()

#         build_info = self.config.build_info
#         storage_path = build_info.storage

#         datasets = dict()

#         if not os.path.exists(storage_path):
#             warnings.warn("storage path {} does not exist.".format(storage_path))

#         # create datasets
#         dataset_cls = self.train_dataset_cls
#         datasets['train'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'train')],
#         )
#         try:
#             datasets['valid'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'valid_small2')])
#             datasets['test'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'test')])
#         except:
#             pass

        

#         return datasets

# @registry.register_builder("amazon")
# class AmazonBuilder(RecBaseDatasetBuilder):
#     train_dataset_cls = AmazonDataset

#     DATASET_CONFIG_DICT = {
#         "default": "configs/datasets/amazon/default.yaml",
#     }

#     def build_datasets(self):
#         # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
#         logging.info("Building datasets...")
#         self.build_processors()

#         build_info = self.config.build_info
#         storage_path = build_info.storage

#         datasets = dict()

#         if not os.path.exists(storage_path):
#             warnings.warn("storage path {} does not exist.".format(storage_path))

#         # create datasets
#         dataset_cls = self.train_dataset_cls
#         datasets['train'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'train')],
#         )
#         try:
#             datasets['valid'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'valid_small')])
#             #0915
#             datasets['test'] = dataset_cls(
#             text_processor=self.text_processors["train"],
#             ann_paths=[os.path.join(storage_path, 'test')])
#         except:
#             print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
#             raise FileNotFoundError("file not found.")
#         return datasets


@registry.register_builder("movie_ood")
class MoiveOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
            if evaluate_only:
                datasets['test_warm'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test=warm')])

                datasets['test_cold'] = dataset_cls(
                text_processor=self.text_processors["train"],
                ann_paths=[os.path.join(storage_path, 'test=cold')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets


@registry.register_builder("movie_ood_sasrec")
class MoiveOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = MoiveOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/movielens/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets



@registry.register_builder("amazon_ood")
class AmazonOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self, evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets with robust existence checks
        dataset_cls = self.train_dataset_cls
        train_prefix = os.path.join(storage_path, 'train')
        valid_small_prefix = os.path.join(storage_path, 'valid_small')
        valid_prefix = os.path.join(storage_path, 'valid')
        test_prefix = os.path.join(storage_path, 'test')

        missing = []
        if not os.path.exists(train_prefix + "_ood2.pkl"):
            missing.append(train_prefix + "_ood2.pkl")
        # choose valid file: prefer valid_small then valid
        if os.path.exists(valid_small_prefix + "_ood2.pkl"):
            chosen_valid = valid_small_prefix
        elif os.path.exists(valid_prefix + "_ood2.pkl"):
            chosen_valid = valid_prefix
        else:
            missing.append(valid_small_prefix + "_ood2.pkl")
            missing.append(valid_prefix + "_ood2.pkl")
            chosen_valid = None
        if not os.path.exists(test_prefix + "_ood2.pkl"):
            missing.append(test_prefix + "_ood2.pkl")

        if missing:
            raise FileNotFoundError(f"AmazonOODBuilder missing files: {missing}")

        datasets['train'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[train_prefix])
        datasets['valid'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[chosen_valid])
        datasets['test'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[test_prefix])

        if evaluate_only:
            # optional splits; only created if base test exists
            warm_prefix = os.path.join(storage_path, 'test=warm')
            cold_prefix = os.path.join(storage_path, 'test=cold')
            if os.path.exists(test_prefix + "_ood2.pkl"):
                datasets['test_warm'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[warm_prefix])
                datasets['test_cold'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[cold_prefix])

        return datasets


@registry.register_builder("amazon_ood_rating")
class AmazonOODBuilder(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData_rating

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self, evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets with robust existence checks
        dataset_cls = self.train_dataset_cls
        train_prefix = os.path.join(storage_path, 'train')
        valid_small_prefix = os.path.join(storage_path, 'valid_small')
        valid_prefix = os.path.join(storage_path, 'valid')
        test_prefix = os.path.join(storage_path, 'test')

        missing = []
        if not os.path.exists(train_prefix + "_ood2.pkl"):
            missing.append(train_prefix + "_ood2.pkl")
        if os.path.exists(valid_small_prefix + "_ood2.pkl"):
            chosen_valid = valid_small_prefix
        elif os.path.exists(valid_prefix + "_ood2.pkl"):
            chosen_valid = valid_prefix
        else:
            missing.append(valid_small_prefix + "_ood2.pkl")
            missing.append(valid_prefix + "_ood2.pkl")
            chosen_valid = None
        if not os.path.exists(test_prefix + "_ood2.pkl"):
            missing.append(test_prefix + "_ood2.pkl")

        if missing:
            raise FileNotFoundError(f"AmazonOODBuilder(rating) missing files: {missing}")

        datasets['train'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[train_prefix])
        datasets['valid'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[chosen_valid])
        datasets['test'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[test_prefix])

        if evaluate_only:
            warm_prefix = os.path.join(storage_path, 'test=warm')
            cold_prefix = os.path.join(storage_path, 'test=cold')
            if os.path.exists(test_prefix + "_ood2.pkl"):
                datasets['test_warm'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[warm_prefix])
                datasets['test_cold'] = dataset_cls(text_processor=self.text_processors["train"], ann_paths=[cold_prefix])

        return datasets


@registry.register_builder("amazon_ood_sasrec")
class AmazonOODBuilder_sasrec(RecBaseDatasetBuilder):
    train_dataset_cls = AmazonOOData_sasrec

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/amazon/default.yaml",
    }
    def build_datasets(self,evaluate_only=False):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'train')],
        )
        try:
            datasets['valid'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'valid_small')])
            #0915
            datasets['test'] = dataset_cls(
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'test')])
        except:
            print(os.path.join(storage_path, 'valid_small'), os.path.exists(os.path.join(storage_path, 'valid_small_seqs.pkl')))
            raise FileNotFoundError("file not found.")
        return datasets
