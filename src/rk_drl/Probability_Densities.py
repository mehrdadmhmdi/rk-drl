import torch
from torch import Tensor
from typing import Optional

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
    Information about the torch-based class to be provided ..... 


'''
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Probability_Densities:
    def __init__(self, **kwargs):
        self.distributions = {}
        supported_pdfs = ['beta', 'gaussian', 'uniform', 'logistic']
        for pdf_choice in supported_pdfs:
            if pdf_choice in kwargs:
                params = kwargs[pdf_choice]
                required_params = {
                    'beta'    : ['theta_alpha', 'theta_beta' ],
                    'gaussian': ['theta_mean' , 'theta_std'  ],
                    'uniform' : ['theta_lower', 'theta_upper'],
                    'logistic': ['theta_loc'  , 'theta_scale']
                }[pdf_choice]
                missing_params = [p for p in required_params if p not in params]
                if missing_params:
                    continue

                epsilon_params = {k.replace('epsilon_', ''): v for k, v in params.items() if k.startswith('epsilon_')}
                for param in required_params:
                    suffix = param.replace('theta_', '')
                    if suffix not in epsilon_params:
                        epsilon_params[suffix] = 0.0


                cleaned_params = {k: v for k, v in params.items() if not k.startswith('epsilon_')}
                self.distributions[pdf_choice] = {
                    'params': cleaned_params,
                    'epsilon': epsilon_params
                }

    # ------------------------------ Helpers -----------------------------------
    @staticmethod
    def _to_tensor(x, like: Optional[Tensor] = None) -> Tensor:
        # Ensure device/dtype match 'like' when provided — avoids CPU/CUDA mismatches
        if isinstance(x, Tensor):
            if like is not None and (x.device != like.device or x.dtype != like.dtype):
                return x.to(device=like.device, dtype=like.dtype)
            return x
        if like is not None:
            return torch.as_tensor(x, dtype=like.dtype, device=like.device)
        return torch.as_tensor(x, dtype=torch.float64)

    @staticmethod
    def _lin(s: Tensor, theta: Tensor) -> Tensor:
        # Works for 1D (d,) • (d,) -> (), and batched (n,d) • (d,) -> (n,)
        return s.matmul(theta)

    @staticmethod
    def _normal_pdf(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        var     = std * std
        log_pdf = -0.5 * torch.log(2 * torch.pi * var) - (x - mean) ** 2 / (2 * var)
        return torch.exp(log_pdf)

    @staticmethod
    def _uniform_pdf(x: Tensor, low: Tensor, high: Tensor) -> Tensor:
        width      = high - low
        in_support = (x >= low) & (x <= high)
        return torch.where(in_support & (width > 0), 1.0 / width, torch.zeros_like(x))

    @staticmethod
    def _logistic_pdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        # f(x) = exp(-(x-loc)/scale) / [ scale * (1 + exp(-(x-loc)/scale))^2 ]
        z  = (x - loc) / scale
        ez = torch.exp(-z)
        denom = scale * (1 + ez) ** 2
        return ez / denom

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pdf(self, pdf_choice, x, s):
        if pdf_choice not in self.distributions:
            # match original: 0.0 for scalar, zeros_like for array
            if not isinstance(x, Tensor):
                return 0.0
            return torch.zeros_like(x, dtype=x.dtype, device=x.device)

        dist_info = self.distributions[pdf_choice]
        params    = dist_info['params']
        epsilon   = dist_info['epsilon']

        try:
            s = self._to_tensor(s)
            x = self._to_tensor(x, like=s)

            # Coerce parameters to tensors on s's device/dtype
            def P(name):
                return self._to_tensor(params[name], like=s)

            if pdf_choice == 'beta':
                theta_a = P('theta_alpha')
                theta_b = P('theta_beta')
                alpha    = torch.exp(self._lin(s, theta_a) + self._to_tensor(epsilon['alpha'], like=s))
                beta_val = torch.exp(self._lin(s, theta_b) + self._to_tensor(epsilon['beta'], like=s))

                # Broadcast to x shape
                alpha_b = alpha if alpha.shape == x.shape else alpha.expand_as(x)
                beta_b  = beta_val if beta_val.shape == x.shape else beta_val.expand_as(x)

                # Beta pdf via torch.special
                eps       = torch.finfo(x.dtype).eps
                x_clamped = torch.clamp(x, eps, 1 - eps)
                log_pdf   = (alpha_b - 1) * torch.log(x_clamped) + (beta_b - 1) * torch.log(1 - x_clamped) \
                          - (torch.lgamma(alpha_b) + torch.lgamma(beta_b) - torch.lgamma(alpha_b + beta_b))
                return torch.exp(log_pdf)

            elif pdf_choice == 'gaussian':
                mean = self._lin(s, P('theta_mean')) + self._to_tensor(epsilon['mean'], like=s)
                std  = torch.exp(self._lin(s, P('theta_std')) + self._to_tensor(epsilon['std'], like=s))

                if torch.any(std <= 0) or torch.any(torch.isnan(std)):
                    print("Invalid negative std detected:", std)

                mean_b = mean if mean.shape == x.shape else mean.expand_as(x)
                std_b  = std  if std.shape  == x.shape else std.expand_as(x)
                pdf_vals = self._normal_pdf(x, mean_b, std_b)
                return torch.sigmoid(pdf_vals)

            elif pdf_choice == 'uniform':
                lower = self._lin(s, P('theta_lower')) + self._to_tensor(epsilon['lower'], like=s)
                upper = self._lin(s, P('theta_upper')) + self._to_tensor(epsilon['upper'], like=s)

                need_fix = upper <= lower
                upper = torch.where(need_fix, lower + 1.0, upper)

                upper = torch.sigmoid(upper)
                lower = torch.sigmoid(lower)

                lower_b = lower if lower.shape == x.shape else lower.expand_as(x)
                upper_b = upper if upper.shape == x.shape else upper.expand_as(x)

                return self._uniform_pdf(x, lower_b, upper_b)

            elif pdf_choice == 'logistic':
                loc   = self._lin(s, P('theta_loc'))   + self._to_tensor(epsilon['loc'], like=s)
                scale = torch.exp(self._lin(s, P('theta_scale')) + self._to_tensor(epsilon['scale'], like=s))
                loc   = torch.clamp(loc, -50.0, 50.0)

                loc_b   = loc   if loc.shape   == x.shape else loc.expand_as(x)
                scale_b = scale if scale.shape == x.shape else scale.expand_as(x)

                pdf_vals = self._logistic_pdf(x, loc_b, scale_b)
                return torch.sigmoid(pdf_vals)

            else:
                return self._to_tensor(0.0, like=s)
        except Exception:
            if not isinstance(x, Tensor):
                return 0.0
            return torch.zeros_like(x)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def sample_pdf(self, pdf_choice, s):
        if pdf_choice not in self.distributions:
            return None

        dist_info = self.distributions[pdf_choice]
        params    = dist_info['params']
        epsilon   = dist_info['epsilon']

        try:
            s = self._to_tensor(s)

            def P(name):
                return self._to_tensor(params[name], like=s)

            if pdf_choice == 'beta':
                alpha    = torch.exp(self._lin(s, P('theta_alpha')) + self._to_tensor(epsilon['alpha'], like=s))
                beta_val = torch.exp(self._lin(s, P('theta_beta')) + self._to_tensor(epsilon['beta'], like=s))
                alpha    = torch.clamp(alpha, min=1e-6).reshape(-1)
                beta_val = torch.clamp(beta_val, min=1e-6).reshape(-1)
                dist     = torch.distributions.Beta(alpha, beta_val)
                sample   = dist.sample()  # (1,) for scalar policy
                return sample

            elif pdf_choice == 'gaussian':
                mean = self._lin(s, P('theta_mean')) + self._to_tensor(epsilon['mean'], like=s)
                std  = torch.exp(self._lin(s, P('theta_std')) + self._to_tensor(epsilon['std'], like=s))
                if torch.any(std <= 0) or torch.any(torch.isnan(std)):
                    print("Invalid negative std detected:", std)
                dist = torch.distributions.Normal(mean, std)
                sample = dist.sample()
                return torch.sigmoid(sample)

            elif pdf_choice == 'uniform':
                lower = self._lin(s, P('theta_lower')) + self._to_tensor(epsilon['lower'], like=s)
                upper = self._lin(s, P('theta_upper')) + self._to_tensor(epsilon['upper'], like=s)
                need_fix = upper <= lower
                upper = torch.where(need_fix, lower + 1.0, upper)
                upper, lower = torch.sigmoid(upper), torch.sigmoid(lower)
                u = torch.rand_like(lower)
                sample = lower + (upper - lower) * u
                return sample

            elif pdf_choice == 'logistic':
                loc   = self._lin(s, P('theta_loc')) + self._to_tensor(epsilon['loc'], like=s)
                loc   = torch.clamp(loc, -50.0, 50.0)
                scale = torch.exp(self._lin(s, P('theta_scale')) + self._to_tensor(epsilon['scale'], like=s))
                u = torch.clamp(torch.rand_like(loc), 1e-6, 1 - 1e-6)
                logistic_sample = loc + scale * torch.log(u / (1 - u))
                return torch.sigmoid(logistic_sample)

            else:
                return None
        except Exception as e:
            print("[sample_pdf error]", pdf_choice, "state:", s, "::", repr(e))
            return None
