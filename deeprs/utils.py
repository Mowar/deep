import json
import logging
from threading import Thread

import requests
from deep.initializers import RandomNormal
from deep.callbacks import Callback
from deep.regularizers import l2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from .activations import *
from .layers import *

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse

custom_objects = {
                  'MLP': MLP,
                  'PredictionLayer': PredictionLayer,
                  'FM': FM,
                  'Dice': Dice}


def get_input(feature_dim_dict, bias_feature_dim_dict=None):
    """define input layer.

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param bias_feature_dim_dict: defautl is None
    """
    # sparse input
    sparse_input = [Input(shape=(1,), name='sparse_' + str(i) + '-' + feat) for i, feat in
                    enumerate(feature_dim_dict["sparse"])]
    # dense input
    dense_input = [Input(shape=(1,), name='dense_' + str(i) + '-' + feat) for i, feat in
                   enumerate(feature_dim_dict["dense"])]

    if bias_feature_dim_dict is None:
        return sparse_input, dense_input
    else:
        bias_sparse_input = [Input(shape=(1,), name='bias_sparse_' + str(i) + '-' + feat) for i, feat in
                             enumerate(bias_feature_dim_dict["sparse"])]

        bias_dense_input = [Input(shape=(1,), name='bias_dense_' + str(i) + '-' + feat) for i, feat in
                            enumerate(bias_feature_dim_dict["dense"])]

        return sparse_input, dense_input, bias_sparse_input, bias_dense_input


def get_share_embeddings(feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    """get share embedding vectors(sparse_embedding and linear_embedding)

    :param feature_dim_dict: dict,to indicate sparse field and dense field like {'sparse':{'field_1':4,'field_2':3,'field_3':2},'dense':['field_4','field_5']}
    :param embedding_size: positive integer,sparse feature embedding_size
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param l2_rev_V: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    """
    # sparse embedding effect
    sparse_embedding = [Embedding(feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(feature_dim_dict["sparse"])]

    # linear embedding effect
    linear_embedding = [Embedding(feature_dim_dict["sparse"][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(feature_dim_dict["sparse"])]

    return sparse_embedding, linear_embedding


def get_sep_embeddings(deep_feature_dim_dict, wide_feature_dim_dict, embedding_size, init_std, seed, l2_rev_V, l2_reg_w):
    sparse_embedding = [Embedding(deep_feature_dim_dict["sparse"][feat], embedding_size,
                                  embeddings_initializer=RandomNormal(
                                      mean=0.0, stddev=init_std, seed=seed),
                                  embeddings_regularizer=l2(l2_rev_V),
                                  name='sparse_emb_' + str(i) + '-' + feat) for i, feat in
                        enumerate(deep_feature_dim_dict["sparse"])]
    linear_embedding = [Embedding(wide_feature_dim_dict["sparse"][feat], 1,
                                  embeddings_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                      seed=seed), embeddings_regularizer=l2(l2_reg_w),
                                  name='linear_emb_' + str(i) + '-' + feat) for
                        i, feat in enumerate(wide_feature_dim_dict["sparse"])]

    return sparse_embedding, linear_embedding


def check_version(version):
    """Return version of package on pypi.python.org using json."""

    def check(version):
        try:
            url_pattern = 'https://pypi.python.org/pypi/DeepRS/json'
            req = requests.get(url_pattern)
            latest_version = parse('0')
            version = parse(version)
            if req.status_code == requests.codes.ok:
                j = json.loads(req.text.encode('utf-8'))
                releases = j.get('releases', [])
                for release in releases:
                    ver = parse(release)
                    if not ver.is_prerelease:
                        latest_version = max(latest_version, ver)
                if latest_version > version:
                    logging.warning('\nDeeprs version {0} detected. Your version is {1}.\nUse `pip install -U DeepRS` to upgrade.Changelog: https://github.com/YSZYCF/DeepRS/releases/tag/v{0}'.format(
                        latest_version, version))
        except Exception:
            return
    Thread(target=check, args=(version,)).start()

class LossHistory(Callback):
    """
        plot metric or loss curve
    """
    def __init__(self, loss_name, metric_name=None):
        """
            define loss function name and metric function name
        """
        self.loss_name = loss_name
        self.metric_name = metric_name

    def on_train_begin(self, logs={}):
        """
        init
        :param logs:
        :return:
        """
        self.train_loss = {'batch':[], 'epoch':[]}
        self.train_metric  = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_metric = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.train_loss['batch'].append(logs.get("loss"))
        self.val_loss['batch'].append(logs.get('val_loss'))

        if self.metric_name != None:
            self.train_metric['batch'].append(logs.get(self.metric_name))
            self.val_metric['batch'].append(logs.get('val_%s' % self.metric_name))

    def on_epoch_end(self, batch, logs={}):
        self.train_loss['epoch'].append(logs.get("loss"))
        self.val_loss['epoch'].append(logs.get('val_loss'))

        if self.metric_name != None:
            self.train_metric['epoch'].append(logs.get(self.metric_name))
            self.val_metric['epoch'].append(logs.get('val_%s' % self.metric_name))

    def loss_plot(self, loss_type="epoch", metric=False):
        """
            plot curve
        :param loss_type:
        :param metric:
        :return:
        """

        if loss_type not in  set(["epoch","batch"]):
            raise ValueError(" loss_type value must be epoch or batch")

        iters = range(len(self.train_loss[loss_type]))

        plt.figure()

        # train loss curve
        plt.plot(iters, self.train_loss[loss_type], 'r', label='train loss')
        # val loss curve
        plt.plot(iters, self.val_loss[loss_type], 'g', label='val loss')

        if (metric == True) and (self.metric_name != None):
            plt.plot(iters, self.train_metric[loss_type], 'b', label='train %s'%(self.metric_name))
            plt.plot(iters, self.val_metric[loss_type], 'k', label='val %'%(self.metric_name))

        plt.grid(True)
        plt.xlabel(loss_type)

        if (metric == True) and (self.metric_name != None):
            plt.ylabel('%s-%s'%(self.loss_name,self.metric_name))
        else:
            plt.ylabel('%s'%(self.loss_name))

        plt.legend(loc="upper right")
        plt.show()
