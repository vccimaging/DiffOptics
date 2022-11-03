from .basics import *
import numpy as np


class Optimization(PrettyPrinter):
    """
    Generic class for optimizers.
    """
    def __init__(self, lens, diff_parameters_names):
        self.lens = lens
        self.diff_parameters_names = []
        self.diff_parameters = []
        
        # TODO: re-sorting names to make sure strings go first
        # diff_parameters_names = sorted(diff_parameters_names, key=lambda x: (x is not None, '' if isinstance(x, Number) else type(x).__name__, x))
        # diff_parameters_names.reverse()
        
        for name in diff_parameters_names:
            if type(name) is str: # lens parameter name
                self.diff_parameters_names.append(name)
                try:
                    exec('self.lens.{}.requires_grad = True'.format(name))
                except:
                    exec('self.lens.{name} = self.lens.{name}.detach()'.format(name=name))
                    exec('self.lens.{}.requires_grad = True'.format(name))
                exec('self.diff_parameters.append(self.lens.{})'.format(name))
            if type(name) is torch.Tensor: # actual parameter
                name.requires_grad = True
                self.diff_parameters.append(name)


class Adam(Optimization):
    """
    Adam gradient descent optimizer.
    """
    def __init__(self, lens, diff_parameters_names, lr, lrs=None, beta=0.99, gamma_rate=None):
        Optimization.__init__(self, lens, diff_parameters_names)
        if lrs is None:
            lrs = [1] * len(self.diff_parameters)
        self.optimizer = torch.optim.Adam(
            [{"params": v, "lr": lr*l} for v, l in zip(self.diff_parameters, lrs)],
            betas=(beta,0.999), amsgrad=True
        )
        if gamma_rate is None:
            gamma_rate = 0.95
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma_rate)

    def optimize(self, loss_func, render=None, maxit=300, record=True):
        print('optimizing ...')
        ls = []
        Is = []
        with torch.autograd.set_detect_anomaly(False): # For speed purposes
            for it in range(maxit):
                if render is None:
                    L = loss_func()
                else:
                    y = render()
                    L = loss_func(y)
                    if record:
                        Is.append(y.cpu().detach().numpy())
                self.optimizer.zero_grad()
                L.backward(retain_graph=True)

                # Report loss
                grads = torch.Tensor([torch.mean(torch.abs(v.grad)) for v in self.diff_parameters])
                print('iter = {}: loss = {:.4e}, grad_bar = {:.4e}'.format(
                    it, L.item(), torch.mean(grads)
                ))
                ls.append(L.cpu().detach().numpy())

                # Your log here:
                # ...

                # Descent
                self.optimizer.step()
                self.lr_scheduler.step()

        return {'ls': np.array(ls), 'Is': np.array(Is)}


class LM(Optimization):
    """
    The Levenberg–Marquardt (LM) algorithm, with the Jacobians evaluated using autodiff.
    For optical designs the classical LM method is a nice feature to have, though it is not
    the primal goal to implement an autodiff optical engine for just computing the Jacobians.
    """
    def __init__(self, lens, diff_parameters_names, lamb, mu=None, option='diag'):
        Optimization.__init__(self, lens, diff_parameters_names)
        self.lamb = lamb # damping factor
        self.mu = 2.0 if mu is None else mu # dampling rate (>1)
        self.option = option
        
    def jacobian(self, func, v=None, return_primal=False):
        """Constructs a M-by-N Jacobian matrix where M >> N.

        Here, computing the Jacobian only makes sense for a tall Jacobian matrix. In this case,
        column-wise evaluation (forward-mode, or jvp) is more effective to construct the Jacobian.

        This function is modified from torch.autograd.functional.jvp().
        """

        Js = []
        outputs = func()

        if v is None:
            v = torch.ones_like(outputs, requires_grad=True)
        else:
            assert v.shape == outputs.shape

        for x in self.diff_parameters:
            N = torch.numel(x)
            vjp = torch.autograd.grad(outputs, x, v, create_graph=True)[0].view(-1)

            if N == 1:
                J = torch.autograd.grad(vjp, v, retain_graph=False, create_graph=False)[0][...,None]
            else:
                I = torch.eye(N, device=x.device)
                J = []
                for i in range(N):
                    Ji = torch.autograd.grad(vjp, v, I[i], retain_graph=True)[0]
                    J.append(Ji.detach().clone())
                J = torch.stack(J, axis=-1)
            del x.grad, v.grad
            Js.append(J)
            torch.cuda.empty_cache() # prevent memory leak
        
        if return_primal:
            return torch.cat(Js, axis=-1), outputs.detach()
        else:
            return torch.cat(Js, axis=-1)

    def optimize(self, func, func_yref_y, maxit=300, record=True):
        """
        Optimization function:

        Inputs:
        - func: Evaluate `y = f(x)` where `x` is the implicit parameters by `self.diff_parameters` (out of the class)
        - func_yref_y: Compute `y_ref - y`

        Outputs:
        - ls: Loss function.
        """
        print('optimizing ...')
        Ns = [x.numel() for x in self.diff_parameters]
        NS = [[*x.shape] for x in self.diff_parameters]

        ls = []
        Is = []
        lamb = self.lamb
        with torch.autograd.set_detect_anomaly(False): # True
            for it in range(maxit):
                y = func()
                Is.append(y.cpu().detach().numpy())
                with torch.no_grad():
                    L = torch.mean(func_yref_y(y)**2).item()
                    if L < 1e-16:
                        print('L too small; termiante.')
                        break

                # obtain Jacobian
                J = self.jacobian(func)
                J = J.view(-1, J.shape[-1])
                JtJ = J.T @ J
                N = JtJ.shape[0]

                # regularization matrix
                if self.option == 'I':
                    R = torch.eye(N, device=JtJ.device)
                elif self.option == 'diag':
                    R = torch.diag(torch.diag(JtJ).abs())
                else:
                    R = torch.diag(self.option)
                
                # compute b = J.T @ (y_ref - y)
                # TODO: inaccurate jvp via pytorch function ... Why?
                # bb = [
                #     torch.autograd.grad(outputs=y, inputs=x, grad_outputs=func_yref_y(y), retain_graph=True)[0]
                #     for x in self.diff_parameters
                # ]
                # breakpoint()
                # for i, bx in enumerate(bb):
                #     if len(bx.shape) == 0: # single scalar
                #         bb[i] = torch.Tensor([bx.item()]).to(y.device)
                #     if len(bx.shape) > 1: # multi-dimension
                #         bb[i] = torch.Tensor(bx.cpu().detach().numpy().flatten()).to(y.device)
                # b = torch.cat(bb, axis=-1)
                # del J, bb, y
                b = J.T @ func_yref_y(y).flatten()

                # damping loop
                L_current = L + 1.0
                it_inner = 0
                while L_current >= L:
                    it_inner += 1
                    if it_inner > 20:
                        print('inner loop too many; Exiting damping loop.')
                        break

                    A = JtJ + lamb * R
                    x_delta = torch.linalg.solve(A, b[...,None])[...,0]
                    if torch.isnan(x_delta).sum():
                        print('x_delta NaN; Exiting damping loop')
                        break
                    x_delta_s = torch.split(x_delta, Ns)

                    # reshape if x is not 1D array
                    x_delta_s = [*x_delta_s]
                    for xi in range(len(x_delta_s)):
                        x_delta_s[xi] = torch.reshape(x_delta_s[xi],  NS[xi])
                    
                    # update `x += x_delta`
                    self.diff_parameters = self._change_parameters(x_delta_s, sign=True)

                    # calculate new error
                    with torch.no_grad():
                        L_current = torch.mean(func_yref_y(func())**2).item()

                    del A

                    # terminate
                    if L_current < L:
                        lamb /= self.mu
                        del x_delta_s
                        break

                    # else, increase damping and undo the update
                    lamb *= 2.0*self.mu
                    # undo x, i.e. `x -= x_delta`
                    self.diff_parameters = self._change_parameters(x_delta_s, sign=False)
                    
                    if lamb > 1e16:
                        print('lambda too big; Exiting damping loop.')
                        del x_delta_s
                        break

                del JtJ, R, b

                # record
                x_increment = torch.mean(torch.abs(x_delta)).item()
                print('iter = {}: loss = {:.4e}, |x_delta| = {:.4e}'.format(
                    it, L, x_increment
                ))
                ls.append(L)
                if it > 0:
                    dls = np.abs(ls[-2] - L)
                    if dls < 1e-8:
                        print("|\Delta loss| = {:.4e} < 1e-8; Exiting LM loop.".format(dls))
                        break

                if x_increment < 1e-8:
                    print("|x_delta| = {:.4e} < 1e-8; Exiting LM loop.".format(x_increment))
                    break
        return {'ls': np.array(ls), 'Is': np.array(Is)}

    def _change_parameters(self, xs, sign=True):
        diff_parameters = []
        for i, name in enumerate(self.diff_parameters_names):
            if sign:
                exec('self.lens.{name} = self.lens.{name} + xs[{i}]'.format(name=name,i=i))
            else:
                exec('self.lens.{name} = self.lens.{name} - xs[{i}]'.format(name=name,i=i))
            exec('diff_parameters.append(self.lens.{})'.format(name))
        for j in range(i+1, len(self.diff_parameters)):
            diff_parameters.append(self.diff_parameters[j] + 2*(sign - 0.5) * xs[j])
        return diff_parameters


class Adjoint(Optimization):
    """
    Adjoint method to compute back-propagation gradients.
    """
    
    def __init__(self, lens, diff_parameters_names, network_func, render_batch_func, paras, verbose=False):
        super().__init__(lens, diff_parameters_names)
        self.network_func = network_func
        self.render_batch_func = render_batch_func
        self.paras = paras
        self.Js = []
        for diff_para in self.diff_parameters:
            if diff_para.dim() == 0: # single-element tensor
                self.Js.append(torch.zeros(1, device=diff_para.device))
            else:
                self.Js.append(torch.zeros(len(diff_para), device=diff_para.device))

        self.verbose = verbose

    def __call__(self):
        """
        This is the core implementation of adjoint backpropagation. Full gradients of the
        differentiable optical parameters are computed in three steps:
        (1) Forward rendering to get the primal, without autodiff.
        (2) Compute the back-propagated gradients from the loss function, usually coupled
        with a custom network.
        (3) Back-propagate the gradients from (2) all the way to the optical parameters.
        """
        # (1) Forward rendering
        I = 0.0
        with torch.no_grad():
            for para in self.paras:
                I += self.render_batch_func(para)
        if self.verbose:
            I_primal = I

        # (2) Compute the back-propagated gradients
        I.requires_grad = True
        L = self.network_func(I) # your network ...
        L.backward()
        I_grad = I.grad
        L_item = L.item()
        del I, L

        # (3) Back-propagate the gradients from (2), and aggregate
        for para in self.paras:
            self._adjoint_batch(self.render_batch_func(para), I_grad)
        torch.cuda.empty_cache()

        # Return
        if self.verbose:
            return L_item, self.Js, I_primal, I_grad
        else:
            return L_item, self.Js

    def _adjoint_batch(self, outputs, adjoint_image):
        for J, x in zip(self.Js, self.diff_parameters):
            vjp = torch.autograd.grad(outputs, x, adjoint_image, retain_graph=True)[0]
            J += vjp.view(-1).detach()
            torch.cuda.empty_cache() # prevent memory leak
            