import sys, time
sys.path.insert(0, "/media/72TB_disk/personal/zk/USSnet/step4_our_new_model_resunet_boundary/trainer")

import torch
from preenc_unet import PreEncUNet

FEATS = [32, 64, 128, 256, 512, 512, 512, 512]
KERN = [[3, 3]] * 8
STRIDES = [[1, 1]] + [[2, 2]] * 7
NCPS = [2] * 8
NCPD = [2] * 7

for fz in (0, 2):
    net = PreEncUNet(input_channels=1, num_classes=2, features_per_stage=FEATS,
                     kernel_sizes=KERN, strides=STRIDES, n_conv_per_stage=NCPS,
                     n_conv_per_stage_decoder=NCPD, deep_supervision=True,
                     backbone="resnet18", pretrained=True, freeze_upto=fz)
    nparam = sum(p.numel() for p in net.parameters())
    ntrain = net.encoder.trainable_parameter_count() + sum(
        p.numel() for n, p in net.named_parameters() if n.startswith("decoder"))
    print("freeze_upto=%d | params %.2f M | trainable %.2f M | %s"
          % (fz, nparam / 1e6, ntrain / 1e6, net.encoder.load_report()))

    dev = torch.device("cuda")
    net = net.to(dev)
    with torch.no_grad():
        net.decoder.deep_supervision = True
        y = net(torch.randn(1, 1, 640, 1024, device=dev))
        print("   DS outputs:", [tuple(t.shape) for t in y])
        net.decoder.deep_supervision = False
        y = net(torch.randn(1, 1, 640, 1024, device=dev))
        print("   no-DS output:", tuple(y.shape))
        net.decoder.deep_supervision = True

    from dynamic_network_architectures.initialization.weight_init import InitWeights_He
    InitWeights_He(1e-2)(net.encoder.proj)
    InitWeights_He(1e-2)(net.encoder.stages_plain)

    opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                          lr=1e-2, momentum=0.99, nesterov=True, weight_decay=3e-5)
    scaler = torch.amp.GradScaler("cuda")
    crit = torch.nn.CrossEntropyLoss()

    def step(bs):
        x = torch.randn(bs, 1, 640, 1024, device=dev)
        tgt = [torch.randint(0, 2, (bs, 640, 1024), device=dev, dtype=torch.long)]
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
            loss = sum(crit(o, t) for o, t in zip(out, tgt))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        return float(loss)

    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        step(4)
    torch.cuda.synchronize()
    s = time.time()
    for _ in range(3):
        step(4)
    torch.cuda.synchronize()
    dt = (time.time() - s) / 3
    print("   batch 4: %.2f s/step -> %.1f s/epoch (250 it) | peak %.1f GB"
          % (dt, dt * 250, torch.cuda.max_memory_allocated() / 1e9))
    del net, opt, scaler
    torch.cuda.empty_cache()
