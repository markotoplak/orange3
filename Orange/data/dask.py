import pickle

import h5py
import dask.array as da
import numpy as np
import pandas

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable


class DaskTable(Table):

    @classmethod
    def from_file(cls, filename, sheet=None):
        """
        Read a data table from a file. The path can be absolute or relative.

        :param filename: File name
        :type filename: str
        :param sheet: Sheet in a file (optional)
        :type sheet: str
        :return: a new data table
        :rtype: Orange.data.Table
        """
        self = cls()

        f = h5py.File(filename, "r")
        X = f['X']
        Y = f['Y']
        if 'W' in f:
            self.W = da.from_array(f['W'])
        else:
            self.W = np.ones((len(X), 0))
        self.X = da.from_array(X)
        self.Y = da.from_array(Y)

        # TODO for now, metas are in memory
        if "metas" in f:
            self.metas = pickle.loads(np.array(f['metas']).tobytes())
        else:
            self.metas = np.ones((len(self.X), 0))

        self.domain = pickle.loads(np.array(f['domain']).tobytes())

        cls._init_ids(self)

        return self

    def has_missing_attribute(self):
        raise NotImplementedError()

    def checksum(self, include_metas=True):
        raise NotImplementedError()

    def _filter_values(self, filter):
        selection = self._values_filter_to_indicator(filter)
        return self[selection.compute()]

    def _compute_basic_stats(self, columns=None,
                             include_metas=False, compute_variance=False):
        rr = []
        stats = []
        if not columns:
            if self.domain.attributes:
                rr.append(dask_stats(self._X,
                                     compute_variance=compute_variance))
            if self.domain.class_vars:
                rr.append(dask_stats(self._Y,
                                     compute_variance=compute_variance))
            if include_metas and self.domain.metas:
                rr.append(dask_stats(self.metas,
                                     compute_variance=compute_variance))
            if len(rr):
                stats = da.concatenate(rr, axis=0)
        else:
            nattrs = len(self.domain.attributes)
            for column in columns:
                c = self.domain.index(column)
                if 0 <= c < nattrs:
                    S = dask_stats(self._X[:, [c]],
                                   compute_variance=compute_variance)
                elif c >= nattrs:
                    if self._Y.ndim == 1 and c == nattrs:
                        S = dask_stats(self._Y[:, None],
                                       compute_variance=compute_variance)
                    else:
                        S = dask_stats(self._Y[:, [c - nattrs]],
                                       compute_variance=compute_variance)
                else:
                    S = dask_stats(self._metas[:, [-1 - c]],
                                   compute_variance=compute_variance)
                stats.append(S)
            stats = da.concatenate(stats, axis=0)
        stats = stats.compute()
        return stats


def dask_stats(X, compute_variance=False):
    is_numeric = np.issubdtype(X.dtype, np.number)

    if X.size and is_numeric:
        nans = da.isnan(X).sum(axis=0).reshape(-1, 1)
        return da.concatenate((
            da.nanmin(X, axis=0).reshape(-1, 1),
            da.nanmax(X, axis=0).reshape(-1, 1),
            da.nanmean(X, axis=0).reshape(-1, 1),
            da.nanvar(X, axis=0).reshape(-1, 1) if compute_variance else \
                da.zeros((X.shape[1], 1)),
            nans,
            X.shape[0] - nans), axis=1)
    else:
        # metas is currently a numpy table
        nans = (pandas.isnull(X).sum(axis=0) + (X == "").sum(axis=0)) \
            if X.size else np.zeros(X.shape[1])
        return np.column_stack((
            np.tile(np.inf, X.shape[1]),
            np.tile(-np.inf, X.shape[1]),
            np.zeros(X.shape[1]),
            np.zeros(X.shape[1]),
            nans,
            X.shape[0] - nans))


def table_to_dask(table, filename):

    with h5py.File(filename, 'w') as f:
        f.create_dataset("X", data=table.X)
        f.create_dataset("Y", data=table.Y)
        f.create_dataset("domain", data=np.void(pickle.dumps(table.domain)))
        f.create_dataset("metas", data=np.void(pickle.dumps(table.metas)))


if __name__ == '__main__':
    # iris = Table("iris.tab")
    # table_to_dask(iris, "iris.hdf5")
    # dt = DaskTable.from_file("iris.hdf5")

    zoo = Table("zoo.tab")
    table_to_dask(zoo, "zoo.hdf5")

    t = DaskTable.from_file("zoo.hdf5")
    print(t.domain)
    print(t.metas)

    def variable(x):
        if x < 10000:
            return ContinuousVariable("Continuous " + str(x))
        return DiscreteVariable("Discrete " + str(x), ["1", "2", "3", "4"])

    if 0:

        domain = Domain([variable(i) for i in range(20000)])
        x = np.c_[np.random.random((20000, 10000)), np.random.randint(4, size=(20000, 10000))]
        bigtable = Table.from_numpy(domain=domain, X=x)
        # print(bigtable.X.shape)
        table_to_dask(bigtable, "t4e6_mixed.hdf5")

        # print(iris[0])
        # print(dt[0])

        # print(dt.X)
        # print(dt)

