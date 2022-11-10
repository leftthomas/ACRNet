import os

import numpy as np

import utils


class SampleDataset:
    def __init__(self, args, mode="both", sampling='random'):
        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling = sampling
        self.num_segments = args.max_seqlen
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join(args.path_dataset, self.dataset_name + "-I3D-JOINTFeatures.npy")
        self.path_to_annotations = os.path.join(args.path_dataset, self.dataset_name + "-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14

        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.batch_size = args.batch_size
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]
        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

        if args.dataset_name == 'ActivityNet1.2':
            self.filter()

    def filter(self):
        new_testidx = []
        for idx in self.testidx:
            feat = self.features[idx]
            if len(feat) > 10:
                new_testidx.append(idx)
        self.testidx = new_testidx

        new_trainidx = []
        for idx in self.trainidx:
            feat = self.features[idx]
            if len(feat) > 10:
                new_trainidx.append(idx)
        self.trainidx = new_trainidx

    def train_test_idx(self):
        if self.dataset_name == 'Thumos14reduced':
            for i, s in enumerate(self.subset):
                if s.decode("utf-8") == "validation":  # Specific to Thumos14
                    self.trainidx.append(i)
                elif s.decode("utf-8") == "test":
                    self.testidx.append(i)
        else:
            for i, s in enumerate(self.subset):
                if s.decode("utf-8") == "training":
                    self.trainidx.append(i)
                elif s.decode("utf-8") == "validation":
                    self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                if self.dataset_name == 'ActivityNet1.2':
                    if self.features[i].sum() == 0:
                        continue
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self, n_similar=0, is_training=True, similar_size=2):
        if is_training:
            idx = []
            # Load similar pairs
            if n_similar != 0:
                rand_classid = np.random.choice(
                    len(self.classwiseidx), size=n_similar
                )
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(
                        len(self.classwiseidx[rid]),
                        size=similar_size,
                        replace=False,
                    )

                    for k in rand_sampleid:
                        idx.append(self.classwiseidx[rid][k])

            # Load rest pairs
            if self.batch_size - similar_size * n_similar < 0:
                self.batch_size = similar_size * n_similar

            rand_sampleid = np.random.choice(
                len(self.trainidx),
                size=self.batch_size - similar_size * n_similar,
            )

            for r in rand_sampleid:
                idx.append(self.trainidx[r])
            feat = []
            for i in idx:
                ifeat = self.features[i]
                if self.sampling == 'random':
                    sample_idx = self.random_sampling(ifeat.shape[0])
                elif self.sampling == 'uniform':
                    sample_idx = self.uniform_sampling(ifeat.shape[0])
                elif self.sampling == "all":
                    sample_idx = np.arange(ifeat.shape[0])
                else:
                    raise AssertionError('Not supported sampling !')
                ifeat = ifeat[sample_idx]
                feat.append(ifeat)
            feat = np.array(feat)
            labels = np.array([self.labels_multihot[i] for i in idx])
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size:]
            return feat, labels, rand_sampleid

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size:]
            return feat, np.array(labs), vn, done

    def random_sampling(self, num_seg):
        sample_idx = np.append(np.arange(self.num_segments) * num_seg / self.num_segments, num_seg)
        for i in range(self.num_segments):
            if int(sample_idx[i]) == int(sample_idx[i + 1]):
                sample_idx[i] = int(sample_idx[i])
            else:
                sample_idx[i] = np.random.randint(int(sample_idx[i]), int(sample_idx[i + 1]))
        return sample_idx[:-1].astype(np.int)

    def uniform_sampling(self, length):
        # because the length may different as these two line codes, make sure batch size == 1 in test mode
        if length <= self.num_segments:
            return np.arange(length).astype(np.int)
        else:
            return np.floor(np.arange(self.num_segments) * length / self.num_segments).astype(np.int)
