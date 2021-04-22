from fastai.vision import *
from .data import ParallelItemList
from .volume import Volume 
from .transform import get_transforms
from .utils import delegates

__all__ = ['get_brats3d_regression']

class BratsItem(Volume):
    def __repr__(self):
        s = self.__class__.__name__
        if self.idx is not None: s += f'[{self.idx}]'
        return s + f' Age:{self.metadata["Age"]:.2f} {tuple(self.shape)}'

    @property
    def data(self): return [self.v.float(), np.float32(self.metadata['Age'])]

    @delegates(Volume.show)
    def show(self, **kwargs):
        super().show(extra_lbl=['Age'], extra_inf=[f'{self.metadata["Age"]:.2f}'], **kwargs)

class BratsItemList(ParallelItemList):
    _item_cls = BratsItem
    def reconstruct(self, t:Tensor): return self._item_cls(t[0], metadata={'Age':t[1]})


def get_brats3d_regression(path:PathOrStr, bs:int=8, valid_pct:float=0.2, csv_file:str='metadata.csv', only_gtr:bool=False)->DataBunch:
    path = Path(path)
    data = BratsItemList.from_paths(path/'data', path/'labels', path/csv_file).filter_na_lbls('surv')
    
    if only_gtr:
        idxs_gtr = data.metadata.index[data.metadata.ResectionStatus=='GTR']
        data.filter_by_idxs(idxs_gtr)

    data = (data.split_by_rand_pct(valid_pct, seed=34)
                .label_from_bcolz('surv')
                .transform(get_transforms(), tfm_y=True)
                .databunch(bs=bs))

    return data

def add_test_set(data:DataBunch, path:PathOrStr, only_survival:bool=False)->DataBunch:
    '''
    only_survival: if True, removes individuals not considered for evaluation.
    '''
    path = Path(path)
    test_data = BratsItemList.from_paths(path/'data', None, path/'metadata.csv')

    if only_survival:
        idxs_gtr = test_data.metadata.index[test_data.metadata.ResectionStatus=='GTR']
        test_data.filter_by_idxs(idxs_gtr)

    test_labels = EmptyLabelList([0] * len(test_data.items))

    test_ds = data.valid_ds.new(test_data, test_labels)
    test_ds.tfm_y = False

    vdl = data.valid_dl
    test_dl = DataLoader(test_ds, vdl.batch_size, shuffle=False, drop_last=False, num_workers=vdl.num_workers)
    test_dl = DeviceDataLoader(test_dl, vdl.device, vdl.tfms, vdl.collate_fn)
    data.label_list.test = test_ds
    data.test_dl = test_dl
    return data
