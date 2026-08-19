# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import os
import time
import copy
import json
import dill
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
import pickle
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from torch_utils.metrics import metric_main



def setup_snapshot_image_grid(training_set, random_seed=0, gw=None, gh=None):
    rnd = np.random.RandomState(random_seed)
    if gw is None:
        gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    if gh is None:
        gh = np.clip(4320 // training_set.image_shape[1], 4, 32)


    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:

        label_groups = dict()
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)


        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])


        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]


    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)



def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)



def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()



def training_loop(
    run_dir                 = '.',      
    training_set_kwargs     = {},
    data_loader_kwargs      = {},
    G_kwargs                = {},
    D_kwargs                = {},
    G_opt_kwargs            = {},
    D_opt_kwargs            = {},
    augment_kwargs          = None,
    loss_kwargs             = {},
    metrics                 = [],
    random_seed             = 0,
    num_gpus                = 1,
    rank                    = 0,
    batch_size              = 4,
    batch_gpu               = 4,
    ema_kimg                = 10,
    ema_rampup              = 0.05,
    G_reg_interval          = 4,
    D_reg_interval          = 16,
    augment_p               = 0,
    ada_target              = None,
    ada_interval            = 4,
    ada_kimg                = 500,
    total_kimg              = 25000,
    kimg_per_tick           = 4,
    image_snapshot_ticks    = 50,
    network_snapshot_ticks  = 50,
    resume_pkl              = None,
    resume_kimg             = 0,
    cudnn_benchmark         = True,
    abort_fn                = None,
    progress_fn             = None,
    restart_every           = -1,
):

    start_time = time.time()
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True
    __RESTART__ = torch.tensor(0., device=device)
    __CUR_NIMG__ = torch.tensor(resume_kimg * 1000, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)
    __AUGMENT_P__ = torch.tensor(augment_p, dtype=torch.float, device=device)
    __PL_MEAN__ = torch.zeros([], device=device)
    best_fid = 9999


    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()


    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()


    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)


    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')

        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        if ckpt_pkl is not None:
            __CUR_NIMG__ = resume_data['progress']['cur_nimg'].to(device)
            __CUR_TICK__ = resume_data['progress']['cur_tick'].to(device)
            __BATCH_IDX__ = resume_data['progress']['batch_idx'].to(device)
            __AUGMENT_P__ = resume_data['progress'].get('augment_p', torch.tensor(0.)).to(device)
            __PL_MEAN__ = resume_data['progress'].get('pl_mean', torch.zeros([])).to(device)
            best_fid = resume_data['progress']['best_fid']       




    if hasattr(G, 'reinit_stem'):
        G.reinit_stem()
        G_ema.reinit_stem()


    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])


    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')


    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)


    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, G_ema=G_ema, D=D, augment_pipe=augment_pipe, **loss_kwargs)
    phases = []
    iter_dict = {'G': 1, 'D': 1}  

    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            for _ in range(iter_dict[name]):
                phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)
            for _ in range(iter_dict[name]):
                phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)


    grid_size = None
    grid_z = None
    grid_c = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)

        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()

        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)


    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)


    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    if num_gpus > 1:
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.broadcast(__AUGMENT_P__, 0)
        torch.distributed.broadcast(__PL_MEAN__, 0)
        torch.distributed.barrier()
    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)
    augment_p = __AUGMENT_P__
    if augment_pipe is not None:
        augment_pipe.p.copy_(augment_p)
    if hasattr(loss, 'pl_mean'):
        loss.pl_mean.copy_(__PL_MEAN__)
    while True:

        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]


        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))


            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)


            if phase.name in ['Dmain', 'Dboth', 'Dreg'] and hasattr(phase.module, 'feature_networks'):
                phase.module.feature_networks.requires_grad_(False)

            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)


            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()


            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))


        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)


        cur_nimg += batch_size
        batch_idx += 1


        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))


        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue


        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))


        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')


        if (rank == 0) and (restart_every > 0) and (time.time() - start_time > restart_every):
            print('Restart job...')
            __RESTART__ = torch.tensor(1., device=device)
        if num_gpus > 1:
            torch.distributed.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f'Process {rank} leaving...')
            if num_gpus > 1:
                torch.distributed.barrier()


        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)


        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):






                    snapshot_data[key] = value
                del value


            if False:
                snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                if rank == 0:
                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)


        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (
                done or cur_tick % network_snapshot_ticks == 0):
            snapshot_pkl = misc.get_ckpt_path(run_dir)

            snapshot_data['progress'] = {
                'cur_nimg': torch.LongTensor([cur_nimg]),
                'cur_tick': torch.LongTensor([cur_tick]),
                'batch_idx': torch.LongTensor([batch_idx]),
                'best_fid': best_fid,
            }
            if augment_pipe is not None:
                snapshot_data['progress']['augment_p'] = augment_pipe.p.cpu()
            if hasattr(loss, 'pl_mean'):
                snapshot_data['progress']['pl_mean'] = loss.pl_mean.cpu()

            with open(snapshot_pkl, 'wb') as f:
                dill.dump(snapshot_data, f)



        if cur_tick and (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                                                      dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)


            snapshot_pkl = os.path.join(run_dir, f'best_model.pkl')
            cur_nimg_txt = os.path.join(run_dir, f'best_nimg.txt')
            if rank == 0:
                if 'fid50k_full' in stats_metrics and stats_metrics['fid50k_full'] < best_fid:
                    best_fid = stats_metrics['fid50k_full']

                    with open(snapshot_pkl, 'wb') as f:
                        dill.dump(snapshot_data, f)

                    with open(cur_nimg_txt, 'w') as f:
                        f.write(str(cur_nimg))

        del snapshot_data


        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None) and \
                    not (phase.start_event.cuda_event == 0 and phase.end_event.cuda_event == 0):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()


        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)


        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break


    if rank == 0:
        print()
        print('Exiting...')


