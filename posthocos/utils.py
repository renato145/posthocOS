import inspect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve as sk_roc_curve, auc
from fastai.basics import *

__all__ = ['classifier_report', 'regression_report', 'plot_cm', 'plot_auc', 'plot_score_dist', 'save_csv',
           'read_csv_files', 'normalize_tensor', 'get_percentile', 'replace_layers', 'replace_layers_types', 'modify_layers', 'modify_layers_types']

def normalize_tensor(x:Tensor)->Tensor:
    x = x-x.min()
    return x/x.max()

def delegates(to=None, keep=False):
    'Decorator: replace `**kwargs` in signature with params from `to` (from: https://www.fast.ai/2019/08/06/delegation)'
    def _f(f):
        if to is None: to_f,from_f = f.__base__.__init__,f.__init__
        else:          to_f,from_f = to,f
        sig = inspect.signature(from_f)
        sigd = dict(sig.parameters)
        k = sigd.pop('kwargs')
        s2 = {k:v for k,v in inspect.signature(to_f).parameters.items()
              if v.default != inspect.Parameter.empty and k not in sigd}
        sigd.update(s2)
        if keep: sigd['kwargs'] = k
        from_f.__signature__ = sig.replace(parameters=sigd.values())
        return f
    return _f

def try_sm(x):
    if x[:10].sum(1).sum().item() != len(x[:10]): x = F.softmax(x, dim=1)
    return x

def classifier_report(y_score:Tensor, y_true:Tensor, figsize:Collection[int]=(14,5), labels:Collection[str]=['survived', 'deceased'],
                      title:Optional[str]=None, lims:Optional[Floats]=None, dist:bool=False, return_auc:bool=True):
    multiclass = len(labels) > 2
    n_axes = 2
    if multiclass: n_axes -= 1
    if dist: n_axes += 1
    fig, axs = plt.subplots(1, n_axes, figsize=figsize)
    if not multiclass: _,auc,th = plot_auc(y_score, y_true, axs[1], labels=labels)
    else: th = 0.5
    _,acc,report = plot_cm(y_score, y_true, ax=axs[0], th=th, labels=labels)
    if dist: _,lim = plot_score_dist(y_score, y_true, axs[-1], labels=labels, lims=lims)
    t = '' if title is None else r'$\bf{' + title + '}$\n'
    suptitle = f'{t}Accuracy: {acc:.4f}'
    if not multiclass: suptitle += f'          AUC: {auc:.4f}\n\n'+report
    fig.suptitle(suptitle, y=1 if multiclass else 1.15, va='center', ha='center',  fontsize=16)
    plt.tight_layout()
    out = axs,lim if dist else axs
    if return_auc: out = (out, auc)
    return out

def regression_report(y_score, y_true, figsize=(12,4), title=None, samples=None):
    t = '' if title is None else r'$\bf{' + title + '}$\n'
    fig,axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[1.5, 1]}, figsize=figsize)
    n = np.arange(len(y_score))
    if (samples is not None) and (len(n) > samples): n = np.random.choice(n, size=samples, replace=False)
    diffs = np.abs(y_score - y_true)
    mae = diffs.mean()
    mse = (diffs**2).mean()
    t += f'MAE={mae:.6f}, MSE={mse:.6f}'
    axs[0].vlines(n, y_true[n], y_score[n], linestyles='dashed', color='darkred', alpha=0.5, label='Error')
    axs[0].scatter(n, y_true[n],  marker='o', alpha=0.5,  label='Labels')
    axs[0].scatter(n, y_score[n], marker='s', alpha=0.75, label='Predictions')
    axs[0].legend()
    sns.kdeplot(to_np(y_true),  shade=True, ax=axs[1], label='Labels')
    sns.kdeplot(to_np(y_score.flatten()), shade=True, ax=axs[1], label='Predictions')
    plt.tight_layout()
    fig.suptitle(t, y=1.05, va='center', ha='center',  fontsize=14)
    return axs,fig

def plot_cm(y_score, y_true, ax=None, th:float=0.5, figsize=(5,4), size=15, labels=['survived', 'deceased']):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f'Confusion matrix (th={th:.2f})', size=15, fontweight='bold')
        ax.set_xlabel('Predicted', size=14)
        ax.set_ylabel('Actual', size=14)
        if len(y_score.shape) > 1:
            y_score = try_sm(tensor(y_score))
            preds = y_score.argmax(1)
        else:
            preds = y_score>th
        accuracy = accuracy_score(y_true, preds)
        report = classification_report(y_true, preds, target_names=[str(lbl) for lbl in labels])
        report = '\n'.join([(' '*i)+e for e,i in zip(report.split('\n')[:4],[12,0,2,0]) if len(e)>0])
        cm = confusion_matrix(y_true, preds)
        ax.imshow(cm, cmap='Blues')
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels, rotation=90)
        thresh = cm.max() / 2.
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, f'{cm[i,j]}', horizontalalignment="center", size=size,
                    color="white" if cm[i,j] > thresh else "black")
        ax.set_ylim(len(labels)-.5,-.5)
    return ax, accuracy, report

def plot_auc(y_score, y_true, ax=None, figsize=(5,4), labels=['survived', 'deceased']):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('False positive ratio (FPR)', size=14)
    ax.set_ylabel('True positive ratio (TPR)', size=14)
    if len(y_score.shape) > 1: y_score = try_sm(tensor(y_score))[:,1]
    fpr,tpr,thresholds = sk_roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    th = thresholds[np.abs(tpr-(1-fpr)).argmin()] # optimal threshold for classification
    ax.plot([0,1], [0,1], linestyle='-', lw=2, color='r', alpha=.6)
    ax.plot(fpr, tpr, lw=3)
    ax.set_title(f'AUC: {roc_auc:.2f}', size=15, fontweight='bold')
    return ax,roc_auc,th

def plot_score_dist(y_score, y_true, ax=None, figsize=(5,4), labels=['survived', 'deceased'], lims=None):
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    ax.set_title('Score distribution', size=15, fontweight='bold')
    ax.set_xlabel('Score', size=14)
    for i,lbl in enumerate(labels):
        pos = y_score[:,i][y_true==i]
        if type(pos) != np.ndarray: pos = to_np(pos)
        if len(np.unique(pos)) == 1:
            ax.axvline(x=pos[0], c=sns.color_palette()[i], label=lbl)
        else:
            sns.kdeplot(pos, color=sns.color_palette()[i], shade=True, ax=ax, label=lbl)
    if lims is not None: ax.set_xlim(lims)
    return ax, ax.get_xbound()

def save_csv(learn:Learner, verbose:bool=True)->None:
    'Saves the current CSVLogger file without overriding past files. Ex: history_0.csv...'
    n = [int(e.stem.split('_')[1]) for e in learn.path.glob(f'{learn.csv_logger.filename}_*.csv')]
    n = 0 if len(n)==0 else max(n)+1
    csv_name = learn.csv_logger.filename+f'_{n}.csv'
    if verbose: print(f'Saving {csv_name!r}...')
    learn.csv_logger.read_logged_file().to_csv(learn.path/(csv_name), index=False)

def _get_csv_files(path:Path, filename:str)->Collection[Path]: return sorted(list(path.glob(f'{filename}_*.csv')))

def _read_csvs(files:Collection[Path], rename_epochs:bool)->DataFrame:
    df = pd.concat([pd.read_csv(f).assign(n=f.stem.split('_')[1]) for f in files], sort=False)
    if rename_epochs:
        df.reset_index(drop=True, inplace=True)
        df['epoch'] = df.index
    return df

def read_csv_files(path, filename:str='history', rename_epochs:bool=True)->DataFrame:
    "Read all csvs into one pd.DataFrame (ex: history_0.csv, history_1.csv, ...)"
    files = _get_csv_files(path, filename)
    return _read_csvs(files, rename_epochs=rename_epochs)

def get_percentile(x:Tensor, perc:float=0.75)->Rank0Tensor:
    x = x.view(-1)
    n = x.numel()
    return x.kthvalue(int(round(n*perc)))[0]

def replace_layers(model:nn.Module, new_layer:Callable, func:Callable)->nn.Module:
    'Recursively replace layers in a model according to a condition `func`.'
    is_sequential = isinstance(model, nn.Sequential)
    it = enumerate(model.children()) if is_sequential else model.named_children()
    for name,layer in it:
        if func(layer):
            if is_sequential: model[name] = new_layer()
            else            : setattr(model, name, new_layer())

        replace_layers(layer, new_layer, func)

    return model

def replace_layers_types(model:nn.Module, new_layer:Callable, replace_types:Collection[nn.Module])->nn.Module:
    'Recursively replace layers in a model according to types `replace_types`.'
    def filter_types(layer): return any(isinstance(layer,o) for o in listify(replace_types))
    return replace_layers(model, new_layer=new_layer, func=filter_types)

def modify_layers(model:nn.Module, modify_func:Callable, func:Callable)->nn.Module:
    'Recursively modify layers in a model according to a condition `func`.'
    is_sequential = isinstance(model, nn.Sequential)
    it = enumerate(model.children()) if is_sequential else model.named_children()
    for name,layer in it:
        if func(layer):
            if is_sequential: model[name] = modify_func(layer)
            else            : setattr(model, name, modify_func(layer))

        modify_layers(layer, modify_func, func)

    return model

def modify_layers_types(model:nn.Module, modify_func:Callable, replace_types:Collection[nn.Module])->nn.Module:
    'Recursively modify layers in a model according to types `replace_types`.'
    def filter_types(layer): return any(isinstance(layer,o) for o in listify(replace_types))
    return modify_layers(model, modify_func=modify_func, func=filter_types)
