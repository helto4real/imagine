#!/usr/bin/env python3
"""Small original sampler histories and prediction algebra; no weights/startup."""
from pathlib import Path
loader = Path(__file__).with_name('reference_shared_schedules.py')
exec(compile(loader.read_text().split('\nwith torch.no_grad():\n', 1)[0], str(loader), 'exec'))
from functools import partial
from itertools import product
from typing import Callable, Union
from tqdm.auto import tqdm
from scipy import integrate
import torchsde
ns.update(integrate=integrate,torchsde=torchsde)
ns.update(Callable=Callable,Union=Union,tqdm=tqdm,nn=torch.nn)
for path in sorted((a.site/'torchsde').rglob('*.py')):
    relative=path.relative_to(a.site).as_posix()
    expected=next(x['sha256']for x in manifest if x['root']=='site'and x['path']==relative)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,relative
    sources[relative]=expected
ns['np']=numpy
definitions('comfy/k_diffusion/deis.py',['edm2t','cal_poly','t2alpha_fn','cal_intergrand','get_deis_coeff_list'])
ns['deis']=NS(get_deis_coeff_list=ns['get_deis_coeff_list'])
definitions('comfy/k_diffusion/sa_solver.py',['compute_exponential_coeffs','compute_simple_stochastic_adams_b_coeffs','compute_stochastic_adams_b_coeffs','get_tau_interval_func'])
ns['sa_solver']=NS(compute_stochastic_adams_b_coeffs=ns['compute_stochastic_adams_b_coeffs'],get_tau_interval_func=ns['get_tau_interval_func'])
definitions('comfy/model_sampling.py', ['reshape_sigma', 'EPS', 'V_PREDICTION', 'V_PREDICTION_DDPM', 'EDM', 'CONST', 'X0', 'IMG_TO_IMG', 'IMG_TO_IMG_FLOW', 'IMG_TO_IMG_VELOCITY', 'COSMOS_RFLOW'])
definitions('comfy/model_patcher.py', ['set_model_options_post_cfg_function'])
ns.update(partial=partial, trange=lambda n, **kw: range(n), utils=NS(append_dims=lambda x, n: x[(...,) + (None,) * (n-x.ndim)]))
ns['comfy'] = NS(model_sampling=NS(CONST=ns['CONST']), model_patcher=NS(set_model_options_post_cfg_function=ns['set_model_options_post_cfg_function']))
names = ['to_d', 'get_ancestral_step', 'default_noise_sampler', 'sigma_to_half_log_snr', 'sample_euler', 'sample_heun', 'sample_heunpp2', 'sample_dpm_2', 'sample_euler_ancestral', 'sample_euler_ancestral_RF', 'sample_dpm_2_ancestral', 'sample_dpm_2_ancestral_RF', 'sample_dpmpp_2m', 'sample_dpmpp_2m_cfg_pp', 'sample_gradient_estimation', 'sample_gradient_estimation_cfg_pp']
names += ['sample_euler_ancestral_cfg_pp', 'sample_euler_cfg_pp', 'sample_dpmpp_2s_ancestral', 'sample_dpmpp_2s_ancestral_RF', 'sample_dpmpp_2s_ancestral_cfg_pp', 'res_multistep', 'sample_res_multistep', 'sample_res_multistep_cfg_pp', 'sample_res_multistep_ancestral', 'sample_res_multistep_ancestral_cfg_pp', 'DDPMSampler_step', 'generic_step_sampler', 'sample_ddpm', 'sample_lcm', 'sample_ipndm']
names += ['linear_multistep_coeff','sample_lms','BatchedBrownianTree','BrownianTreeNoiseSampler','half_log_snr_to_sigma','offset_first_sigma_for_snr','sample_dpmpp_sde','sample_dpmpp_sde_gpu','sample_dpmpp_2m_sde','sample_dpmpp_2m_sde_gpu','sample_dpmpp_2m_sde_heun','sample_dpmpp_2m_sde_heun_gpu','sample_dpmpp_3m_sde','sample_dpmpp_3m_sde_gpu']
names += ['ei_h_phi_1','ei_h_phi_2','sample_seeds_2','sample_seeds_3','sample_exp_heun_2_x0','sample_exp_heun_2_x0_sde','sample_er_sde']
names += ['sample_ipndm_v','_sample_cfgpp_history','sample_cfgpp_ud10_ab','sample_deis']
names += ['PIDStepSizeController','DPMSolver','sample_dpm_fast','sample_dpm_adaptive','sample_sa_solver','sample_sa_solver_pece']
definitions('comfy/k_diffusion/sampling.py', names)
definitions('comfy/extra_samplers/uni_pc.py', ['NoiseScheduleVP','model_wrapper','UniPC','interpolate_fn','expand_dims','SigmaConvert','predict_eps_sigma','sample_unipc','sample_unipc_bh2'])
ns['sample_uni_pc']=ns['sample_unipc']
ns['sample_uni_pc_bh2']=ns['sample_unipc_bh2']
original_brownian=ns['BrownianTreeNoiseSampler']
algebra = []
solvers = []

def data(shape, device, dtype, offset=0):
    return (((torch.arange(math.prod(shape), device=device, dtype=torch.float32)+offset)%29)*.03125-.40625).reshape(shape).to(dtype)

class OracleBudgetError(RuntimeError):
    """A fixture-harness failure, never a reference algorithm exception."""
class Model:
    def __init__(self, sampling, calls, float_output):
        self.sampling, self.calls, self.float_output = sampling, calls, float_output
        self.inner_model = NS(inner_model=NS(model_sampling=sampling), model_patcher=NS(get_model_object=lambda name: sampling))
    def __call__(self, x, sigma, **kwargs):
        if len(self.calls)>=512:raise OracleBudgetError('Bounded sampler fixture exceeded 512 model calls')
        self.calls.append(dict(x=stamp(x), sigma=stamp(sigma), cfg_pp=kwargs.get('model_options', {}).get('disable_cfg1_optimization', False)))
        y = x*.125 + sigma.reshape(2, 1, 1, 1)*.25
        if self.float_output == 'preserve': y=y.to(x.dtype)
        elif self.float_output: y = y.float()
        for hook in kwargs.get('model_options', {}).get('sampler_post_cfg_function', []):
            y = hook(dict(denoised=y, uncond_denoised=y*.5-.0625,cond_denoised=y*1.25+.03125,uncond='own-negative'))
        return y

with torch.no_grad():
    for device in ['cpu', 'cuda:0']:
        for dtype in [torch.float32, torch.bfloat16]:
            for sigma_shape in [[], [1], [2]]:
                sigma = torch.tensor(.75 if not sigma_shape else ([.75] if sigma_shape == [1] else [.75, .25]), device=device)
                for name in ['EPS', 'V_PREDICTION', 'V_PREDICTION_DDPM', 'EDM', 'CONST', 'X0', 'IMG_TO_IMG', 'IMG_TO_IMG_FLOW', 'IMG_TO_IMG_VELOCITY', 'COSMOS_RFLOW']:
                    model = ns[name]()
                    model.sigma_data = .5
                    model.noise_scale = .875
                    x = data((2,1,2,3), device, dtype, 3)
                    output = data(x.shape, device, dtype, 7)
                    noise = data(x.shape, device, dtype, 5)
                    algebra.append(dict(name=name, device=device, kind=str(dtype), sigma_shape=sigma_shape, input=stamp(model.calculate_input(sigma, x)), denoised=stamp(model.calculate_denoised(sigma, output, x)), noise=stamp(model.noise_scaling(sigma, noise, x)), noise_max=stamp(model.noise_scaling(sigma, noise, x, max_denoise=True)), inverse=stamp(model.inverse_noise_scaling(sigma, x))))
            for flow in [False, True]:
                sampling = ns['CONST']() if flow else ns['EPS']()
                sampling.noise_scale = .875
                sampling.percent_to_sigma=ns['ModelSamplingFlux'](NS(sampling_settings=dict(shift=.69))).percent_to_sigma
                for float_output in [False, True, 'preserve']:
                    for name in ['euler', 'heun', 'heunpp2', 'dpm_2', 'euler_ancestral', 'dpm_2_ancestral', 'dpmpp_2m', 'dpmpp_2m_cfg_pp', 'gradient_estimation', 'gradient_estimation_cfg_pp', 'euler_ancestral_cfg_pp', 'euler_cfg_pp', 'dpmpp_2s_ancestral', 'dpmpp_2s_ancestral_cfg_pp', 'res_multistep', 'res_multistep_cfg_pp', 'res_multistep_ancestral', 'res_multistep_ancestral_cfg_pp', 'ddpm', 'lcm', 'ipndm','lms','dpmpp_sde','dpmpp_sde_gpu','dpmpp_2m_sde','dpmpp_2m_sde_gpu','dpmpp_2m_sde_heun','dpmpp_2m_sde_heun_gpu','dpmpp_3m_sde','dpmpp_3m_sde_gpu','seeds_2','seeds_3','exp_heun_2_x0','exp_heun_2_x0_sde','er_sde','ipndm_v','cfgpp_ud10_ab','deis','sa_solver','sa_solver_pece','dpm_fast','dpm_adaptive','uni_pc','uni_pc_bh2']:
                        options = [dict(s_churn=0.), dict(s_churn=.3, s_tmin=.2, s_tmax=.95, s_noise=.875)] if name in ['euler', 'heun', 'heunpp2', 'dpm_2'] else ([dict(eta=0., s_noise=.875), dict(eta=1., s_noise=.875)] if 'ancestral' in name else ([{},dict(s_noise=.875,s_noise_end=.25,noise_clip_std=1.5)] if name=='lcm' else [{}]))
                        if '_sde' in name and name!='er_sde':options=[dict(eta=0.,s_noise=.875),dict(eta=1.,s_noise=.875)]
                        if name=='seeds_2':options=[dict(eta=0.,s_noise=.875,r=.5,solver_type='phi_1'),dict(eta=1.,s_noise=.875,r=.5,solver_type='phi_1'),dict(eta=1.,s_noise=.875,r=.75,solver_type='phi_2')]
                        if name=='seeds_3':options=[dict(eta=0.,s_noise=.875),dict(eta=1.,s_noise=.875)]
                        if name=='er_sde':options=[dict(s_noise=0.,max_stage=1),dict(s_noise=.875,max_stage=2),dict(s_noise=.875,max_stage=3)]
                        if name in ['sa_solver','sa_solver_pece']:options=[{},dict(predictor_order=2,corrector_order=2,simple_order_2=True)]
                        if name in ['dpm_fast','dpm_adaptive']:options=[dict(eta=0.,s_noise=.875),dict(eta=.5,s_noise=.875)]
                        if name.startswith('dpmpp_')and '_sde'in name:options += [dict(option,_custom=True)for option in options[:]]
                        for options in options:
                            for first,end in product([.9,1.]if ('_sde'in name or name.startswith('seeds_') or name=='exp_heun_2_x0')else[.9],[0.,.05]):
                                calls, callbacks, noises = [], [], []
                                sigmas = torch.tensor([first, .6, .3, .1, end], device=device)
                                x = data((2,1,2,3), device, dtype, 3)
                                model = Model(sampling, calls, float_output)
                                class RecordingBrownian:
                                    def __init__(self,*args,**kwargs):
                                        self.cpu=kwargs.get('cpu',False)
                                        self.source=original_brownian(*args,**kwargs)
                                    def __call__(self,sigma,next):
                                        output=self.source(sigma,next)
                                        noises.append(dict(kind='brownian_cpu'if self.cpu else'brownian_gpu',sigma=stamp(sigma),next=stamp(next),output=stamp(output)))
                                        return output
                                ns['BrownianTreeNoiseSampler']=RecordingBrownian
                                fn = ns['sample_'+name]
                                kwargs = dict(options)
                                custom=kwargs.pop('_custom',False)
                                if custom or 'ancestral' in name or name in ['res_multistep','res_multistep_cfg_pp','ddpm','lcm','seeds_2','seeds_3','exp_heun_2_x0_sde','er_sde','sa_solver','sa_solver_pece','dpm_fast','dpm_adaptive']:
                                    source_noise = ns['default_noise_sampler'](x, seed=901)
                                    def independent(sigma, next):
                                        output = source_noise(sigma, next)
                                        noises.append(dict(kind='custom'if custom else'independent', sigma=stamp(sigma), next=stamp(next), output=stamp(output)))
                                        return output
                                    kwargs['noise_sampler'] = independent
                                original = torch.randn_like
                                def randn_like(t):
                                    y = original(t)
                                    noises.append(dict(kind='global_like', output=stamp(y)))
                                    return y
                                torch.manual_seed(17)
                                torch.randn_like = randn_like
                                failure={}
                                try:
                                    args=(model,x,sigmas)
                                    if name in ['dpm_fast','dpm_adaptive']:
                                        minimum=sigmas[-1]if sigmas[-1]!=0 else sigmas[-2]
                                        args=(model,x,minimum,sigmas[0])
                                        if name=='dpm_fast':args+=(len(sigmas)-1,)
                                    y = fn(*args, extra_args=dict(seed=901), callback=lambda c: callbacks.append({k: (stamp(v) if isinstance(v, torch.Tensor) else v) for k,v in c.items()}), disable=True, **kwargs)
                                except Exception as error:
                                    if isinstance(error,OracleBudgetError):raise
                                    reject_resource_failure(error)
                                    failure=dict(error=type(error).__name__,message=str(error))
                                finally: torch.randn_like = original
                                state = torch.get_rng_state() if device == 'cpu' else torch.cuda.get_rng_state(device)
                                solvers.append(dict(name=name, device=device, kind=str(dtype), flow=flow, float_output=float_output, options=options, first=first, end=end, calls=calls, callbacks=callbacks, noises=noises, state=stamp(state),post_sigmas=stamp(sigmas),**(failure if failure else dict(output=stamp(y)))))
a.output.write_text(json.dumps(dict(schema=1, sources=sources, algebra=algebra, solvers=solvers), separators=(',', ':'))+'\n')
print(f'Captured {len(algebra)} algebra cases and {len(solvers)} full sampler histories; model weights allocated: 0')
