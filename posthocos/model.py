from fastai.vision import *
from fastai.callbacks import *
from .layers import conv_pool
from .utils import delegates, replace_layers_types

BinType = Enum('BinType', 'Uniform Percentiles')

class SimpleBody(nn.Sequential):
    def __init__(self, vol_size:Collection[int], num_layers:int, ni:int=4, nf:int=16, rm_pools:int=0):
        vol_size = listify(vol_size, 3)
        strides = [2 if e > max(vol_size)//2 else 1 for e in vol_size]
        layers = [conv_pool(ni, nf, stride=strides)]
        x = dummy_eval(layers[0].to(defaults.device), vol_size).detach()
        
        for i in range(num_layers-1):
            dims = x.shape[-3:]
            strides = 1 if i>=num_layers-2 else [2 if e > max(dims)//2 else 1 for e in dims]
            layers.append(conv_pool(nf, nf*2, stride=strides))
            nf *= 2
            x = layers[-1].to(defaults.device).eval()(x)
        
        self.num_features = nf
        super().__init__(*layers)

class LSEPool3d(Module):
    def __init__(self, r0:float=0.0):
        '3d version of LSE pooling'
        self.r0 = r0
        self.beta = nn.Parameter(tensor([0.]))

    @property
    def r(self)->float:
        with torch.no_grad(): r = self.beta.exp().add(self.r0).item()
        return r

    def __repr__(self)->str: return f'{self.__class__.__name__} (r0={self.r0:.2f}, beta={self.beta.item():.4f} -> r={self.r:.4f})'

    def forward(self, x):
        r = self.beta.exp().add(self.r0)
        return lse_pool3d(x, r)

def lse_pool3d(x:Tensor, r:float=1.0)->Tensor:
    h,w,d = x.shape[-3:]
    theta = tensor(h*w*d).float().to(x.device).log()
    return x.mul(r).logsumexp((-3,-2,-1)).sub(theta).div(r)

def relu2leakyrelu(model:nn.Module)->nn.Module:
    replace_layers_types(model, partial(nn.LeakyReLU, negative_slope=0.1, inplace=True), nn.ReLU)
    return model

class Saliency3d(Module):
    def __init__(self, body:nn.Module, c:int):
        "Gets the saliency model"
        self.body = body
        self.head = nn.Conv3d(body.num_features, c, 3, padding=1)
        self.pool = LSEPool3d()
        self.age_fts = nn.Sequential(*bn_drop_lin(1, body.num_features))
        relu2leakyrelu(self)

    def get_saliency_layer(self)->nn.Module: return self.head

    def forward(self, vol:Tensor, age:Tensor=None)->Tensor:
        vol_fts = self.body(vol)
        age_fts = self.age_fts(age[:,None])[...,None,None,None]
        out = self.pool(self.head(vol_fts + age_fts))
        return out

def mae(input:Tensor, targs_bbs:Collection[Tensor], targs_lbls:Collection[Tensor], pre_proc:Optional[Callable]=None)->Rank0Tensor:
    if pre_proc is not None: input = pre_proc(input)
    return mean_absolute_error(input, targs_lbls)

def acc(input:Tensor, targs_bbs:Collection[Tensor], targs_lbls:Collection[Tensor], pre_proc:Optional[Callable]=None)->Rank0Tensor:
    if pre_proc is not None: input = pre_proc(input)
    true_lbl = pd.cut(targs_lbls.view(-1).cpu(), [0, 10*30, 15*30, np.inf], labels=['short', 'mid', 'long'])
    pred_lbl = pd.cut(     input.view(-1).cpu(), [0, 10*30, 15*30, np.inf], labels=['short', 'mid', 'long'])
    return tensor((true_lbl==pred_lbl).mean())

class RegressionLoss(Module):
    def __init__(self): self.loss = MSELossFlat()
    def forward(self, input:Tensor, targs_bbs:Tensor, targs_lbls:Tensor)->Tensor: return self.loss(input.flatten(), targs_lbls.flatten())

@delegates(Learner.__init__)
def regression_saliency3d(data:DataBunch, num_layers:int=4, ni:int=4, nf:int=16, c:int=1, **kwargs:Any)->Learner:
    "Regression with saliency map for 3D volumes."
    vol_size = data.x[0].shape[-3:]
    body = SimpleBody(vol_size, num_layers, ni=ni, nf=nf)
    model = Saliency3d(body, c=c)

    # Set Metrics:
    metrics = [mae, acc]

    # Build Learner
    opt_func = partial(optim.Adam, eps=0.1, betas=(0.9,0.99))
    learn = Learner(data, model, metrics=metrics, opt_func=opt_func, **kwargs)
    learn.loss_func = RegressionLoss()

    return learn

def get_bin_weights(x: np.ndarray, bins:int, bin_type:BinType, ll:float=0, ul:float=1800):
    if   bin_type == BinType.Uniform: cuts = pd.cut(x, bins=bins)
    elif bin_type == BinType.Percentiles: cuts = pd.qcut(x, q=bins)
    else: raise Exception(f'Invalid `bin_type`: {bin_type!r}')
        
    c = [o.left for o in cuts.categories[::-1]]
    first = ul-c[0]
    middle = [b-a for a,b in zip(c[1:],c[:-1])]
    last = middle[-1] + c[-1]
    c = [first, *middle[:-1], last]
    return tensor(c).float()

class MixModel(Module):
    def __init__(self, m:nn.Module, bin_weights:FloatTensor):
        self.m,self.bin_weights = m,bin_weights
        self.base = bin_weights.sum()

    def forward(self, *x):
        return self.base - self.m(*x).sigmoid().mul(self.bin_weights).sum(dim=1)

class MixLoss(RegressionLoss):
    def __init__(self, logits_layer:nn.Module, penalty:float=1.0):
        super().__init__()
        self.metric_names = ['main_loss','penalty']
        self.penalty = penalty
        self.hook = hook_output(logits_layer, detach=False)

    def __repr__(self)->str: return f'{self.__class__.__name__}(loss={self.loss}, penalty={self.penalty:.2f})'
        
    def forward(self, input:Tensor, targs_bbs:Tensor, targs_lbls:Tensor)->Tensor:
        'input = ([bbs],[lbls])'
        logits = self.hook.stored.sigmoid()
        losses = [
            self.loss(input.flatten(), targs_lbls.flatten()),
            (logits[:,1:] - logits[:,:-1]).relu().sum(dim=1).mul(self.penalty).mean()
        ]
        self.metrics = dict(zip(self.metric_names, losses))
        return sum(losses)

    def __del__(self)->None: self.hook.remove()

class ParallelLossMetrics(callbacks.LossMetrics):
    'Modification for Parallel models'
    def on_batch_end(self, last_target, train, **kwargs):
        "Update the metrics if not `train`"
        if train: return
        bs = last_target[0].size(0)
        for name in self.names:
            self.metrics[name] += bs * self.learn.loss_func.metrics[name].detach().cpu()
        self.nums += bs

@delegates(regression_saliency3d)
def sigmix_learner(data:DataBunch, bins:int=5, bin_type:BinType=BinType.Uniform, loss_penalty:float=10000, **kwargs)->Learner:
    # Data
    train_lbls = np.asarray(data.train_ds.y.labels)[data.train_ds.y.items]
    if type(bin_type) is str: bin_type = BinType[bin_type]
    bin_weights = get_bin_weights(train_lbls, bins, bin_type).to(data.device)

    # Model
    learn = regression_saliency3d(data, c=bins, **kwargs)
    learn.model = MixModel(learn.model, bin_weights)

    # Loss
    loss_func = MixLoss(learn.model.m, penalty=loss_penalty)
    learn.loss_func = loss_func
    learn.callback_fns.append(ParallelLossMetrics)

    return learn
