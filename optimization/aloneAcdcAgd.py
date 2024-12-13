"""
Exactly like the official SGD optimizer but lets you zero out momentum.
"""

import torch
from torch.optim import Optimizer
import torch.nn.functional as F


class aloneAcdcAgd(Optimizer):

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
        super(aloneAcdcAgd, self).__init__(params, defaults)
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        #abort()

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

        
        print("this is the standalone version")

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
        super(aloneAcdcAgd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)


    ########################

    @torch.no_grad()
    def step(self, closure=None):

        # TO-DO decide what stage we are in 
        phase = self.getCurrentPhase()

        self.iteration += 1
        print(self.iteration)
        print('does this print')


        with torch.no_grad():
            for p in self.paramsIter():       


                # TO-DO: 
                # -modify sparsify, refreeze, freeze


                state = self.state[p]

                state['zt_oldGrad'] = p.grad.clone().detach()
                state['zt_old'] = p.data.clone().detach()

                # First get x_t
                state['xt'] = state['xt'] - (1.0 / pow(self.alpha*self.beta , 0.5)) * p.grad

                # Truncate x_t
                if phase == "compressed" or phase == "fine-tuning":
                    if self.notFrozenYet == True:
                        # Freeze X_T
                        self.freeze(iterate='xt')
                        print('hello')
                    else:
                        # Re-freeze ZT
                        self.refreeze(iterate='xt')

                # Find the new z_t
                ztPlus = (state['zt_old'] - (state['zt_oldGrad'] / self.beta) )
                p.data = (self.sqKappa / (self.sqKappa + 1.0) ) * ztPlus + (1.0 / (self.sqKappa + 1.0)) * state['xt']


                ##################################################### end of iteration


    #########################  SPARSIFICATION FUNCTIONS  ###########################
    def getCutOff(self,sparsity=None,iterate=None):
        if sparsity == None:
            sparsity = self.sparsity

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
                print("!!!!!!!!!!! this should sparsify the params, IN THIS CASE X_T!!!!")
  
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


    #########################################################################################          
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


    ###################################  UTILITY FUNCTIONS  ################################################

    """ Desc: when we add extra kwargs that aren't recognized, we add them to our variables by default"""
    def dealWithKwargs(self,keywordArgs):
        for key, value in keywordArgs.items():
            setattr(self, key, value)

    """ Desc: use it like 'for i in paramsIterator():' """

    def paramsIter(self):
        for group in self.param_groups:
            for p in group['params']:
                yield p





