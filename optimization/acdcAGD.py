"""
Exactly like the official SGD optimizer but lets you zero out momentum.
"""

import torch
from torch.optim import Optimizer
import torch.nn.functional as F


class acdcAGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.
        Considering the specific case of Momentum, the update can be written as
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}
        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the 
        parameters, gradient, velocity, and momentum respectively.
        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form
        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}
        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(acdcAGD, self).__init__(params, defaults)
        self.initial_lr = lr
        self.secretname = "iht_agd"


        self.beta = 10.0
        self.kappa = 10.0

        #self.sparsifyInterval = sparsifyInterval
        self.specificSteps = 0


        # ###
        self.iteration = 0
        self.trialNumber = None
        self.testAccuracy = None

        self.loggingInterval = 100

        # Varaibles specific to certain classes
        self.currentDataBatch = None

        #self.dealWithKwargs(kwargs)

        # Compression, Decompression and Freezing Variables

        ## CIFAR10
        self.phaseLength = 10
        self.compressionRatio = 0.5
        self.freezingRatio = 0.2
        self.warmupLength = 6
        self.startFineTune = 50

        self.methodName = "iht_AGD"
        self.alpha = self.beta / self.kappa

        self.specificSteps = 0

        self.areWeCompressed = False
        self.notFrozenYet = True

        self.batchIndex = 0

        # State Initialization
        for p in self.paramsIter():
            state = self.state[p]
            state['xt_frozen'] = torch.ones_like(p)
            state['xt_gradient'] = torch.zeros_like(p)


        # Objective Function Property Variables
        self.alpha = self.beta / self.kappa
        self.sqKappa = pow(self.kappa,0.5)
        self.loss_zt = 0.0


        for p in self.paramsIter():
            state = self.state[p]

            state['zt'] = torch.zeros_like((p.to(self.device)))
            state['xt'] = p.data.detach().clone()
            state['zt_oldGrad'] = torch.zeros_like((p.to(self.device)))

    @torch.no_grad()
    def reset_momentum_buffer(self):
        print("resetting momentum")
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    continue
                param_state['momentum_buffer'].mul_(0.)

    def __setstate__(self, state):
        super(acdcAGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    ########################

    @torch.no_grad()
    def step(self, closure=None):

        # TO-DO decide what stage we are in 
        phase = self.getCurrentPhase()

        # check how far along we are
        howFarAlong = ((self.iteration - self.warmupLength) % self.phaseLength) + 1

        with torch.no_grad():
            for p in self.paramsIter():                

                state = self.state[p]

                # First Get z_t+
                state['zt'] = (state['zt'] - (state['zt_oldGrad'] / self.beta) )

                # Truncating or Freezing ZT - only happens at the very end
                if phase == "finetuning":
                    if self.notFrozenYet == True:
                        # Freeze ZT
                        self.freeze(iterate='zt')
                    else:
                        # Re-freeze ZT
                        self.refreeze(iterate='zt')

                # After this line, 'zt' is actually 'z_t+' now
                state['zt'] = (self.sqKappa / (self.sqKappa + 1.0) ) * state['zt'] + (1.0 / (self.sqKappa + 1.0)) * state['xt']

            # Computing loss and gradients on 'zt'
            self.getNewGrad('zt')

            with torch.no_grad():
                for p in self.paramsIter():
                    state = self.state[p]

                     # NOTE: p.grad is now the gradient at zt
                    state['zt_oldGrad'] = p.grad.clone().detach()
                    
                    # This is the actual line for updating the network parameters
                    p.data = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad

            # We need to keep a separate storage of xt because we will replace the actual network parameters to get the zt gradient
            self.copyXT()

            if phase == "compressed":
                if self.notFrozenYet:
                    # Truncate and Freeze XT
                    self.sparsify()
                    self.copyXT()
                    self.freeze()
                else:
                    # Re-freeze XT
                    self.refreeze()


    ###### SPARSIFICATION FUNCTIONS
    def getCutOff(self,sparsity=None,iterate=None):
        if sparsity == None:
            sparsity = self.sparsity
        if iterate == 'zt':
            sparsity = 0.95

        concatWeights = torch.zeros((1)).to(self.device)
        for p in self.paramsIter():
            if iterate == None:
                layer = p.data
            else:
                state = self.state[p]
                layer = state[iterate]

        flatWeights = torch.flatten(layer)
        concatWeights = torch.cat((concatWeights,flatWeights),0)
        concatWeights = concatWeights[1:] # Removing first zero

        # Converting the sparsity factor into an integer of respective size
        topK = int(len(concatWeights)*(1-sparsity))

        # All the top-k values are sorted in order, we take the last one as the cutoff
        vals, bestI = torch.topk(torch.abs(concatWeights),topK,dim=0)
        cutoff = vals[-1]

        return cutoff
  
    def sparsify(self,iterate=None):
        cutoff = self.getCutOff(iterate=iterate)

        for p in self.paramsIter():
            state = self.state[p]
            if iterate == None:
                print("!!!!!!!!!!! this should sparsify the params")
                p.data[torch.abs(p) <= cutoff] = 0.0
            else:
                (state[iterate])[torch.abs(state[iterate]) <= cutoff] = 0.0
  
    def refreeze(self,iterate=None):
        for p in self.paramsIter():
            state = self.state[p]
            if iterate == None:
                p.data *= state['xt_frozen']
            else:
                state[iterate] *= state['xt_frozen']

    def freeze(self,iterate=None):
        cutOff = self.getCutOff(iterate=iterate)
        for p in self.paramsIter():
            state = self.state[p]
            if iterate == None:
                layer = p.data
                state['xt_frozen'] = (torch.abs(layer) > 0).type(torch.uint8)
            else:
                layer = state[iterate]
                state[f"{iterate}_frozen"] = (torch.abs(layer) > 0).type(torch.uint8)


    ###### UTILITY FUNCTIONS
    """Expected phases: warmup, compressed, dense, fine-tuning"""
    def getCurrentPhase(self):
        howFarAlong = ((self.iteration - self.warmupLength) % self.phaseLength) + 1

        if self.iteration < self.warmupLength:
            phase  = "warmup"
        elif self.iteration >= self.startFineTune:
            phase = "finetuning"
        elif howFarAlong <= self.phaseLength * self.compressionRatio:
            phase = "compressed"
        elif howFarAlong > self.phaseLength * self.compressionRatio:
            phase = "dense"
        else:
            print("Error, iteration logic is incorrect")

       
    def getNewGrad(self,iterate):
        with torch.no_grad():
            for p in self.paramsIter():
                state = self.state[p]
                p.data = state[iterate].clone().detach()

                self.zero_grad()
                data,target = self.currentDataBatch

                newOutput = self.model(data)
                #criterion = torch.nn.CrossEntropyLoss()
                #loss = criterion(newOutput,target)
                loss = F.cross_entropy(newOutput, target, reduction='sum').item()
                # ACDC loss!!! F.cross_entropy(output, target, reduction='sum').item()
                #loss = F.nll_loss(newOutput, target)
                loss.backward()

    def copyXT(self):
        with torch.no_grad():
            for p in self.paramsIter():
                state = self.state[p]
                state['xt'] = p.data.clone().detach()

    """ Desc: when we add extra kwargs that aren't recognized, we add them to our variables by default"""
    def dealWithKwargs(self,keywordArgs):
        for key, value in keywordArgs.items():
            setattr(self, key, value)

    """ Desc: use it like 'for i in paramsIterator():' """

    def paramsIter(self):
        for group in self.param_groups:
            for p in group['params']:
                yield p





