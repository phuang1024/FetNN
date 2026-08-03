"""Annotated INN training function.
Batch size: 1600

Ground truth and random z:
    X: [x, "zero"]
    Y: [z, "zero", y]
Both ndim_tot

Model forward prediction:
    Y_hat = f(X)
    Y_hat: [z_hat, zero_hat, y_hat]

Ly and Lz:
    Ly = loss(
        (zero_hat, y_hat),
        ("zero", y),
    )
    Lz = MMD(
        (z_hat, y_hat),
        (z, y),
    )

Model backward prediction:
    X_hat = f-1(Y)

Lx and Ly2:
    Lx = MMD(

    )
"""

def train(i_epoch=0):
    model.train()

    l_tot = 0
    batch_idx = 0
    
    t_start = time()
    
    loss_factor = min(1., 2. * 0.002**(1. - (float(i_epoch) / n_epochs)))

    # x: (dim_x), 2 in example.
    # y: (dim_y), 8 in example (one hot).
    for x, y in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break

        x, y = x.to(device), y.to(device)
        
        y_clean = y.clone()
        pad_x = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)

        y += y_noise_scale * torch.randn(batch_size, ndim_y, dtype=torch.float, device=device)

        # Create padded and concat inputs and outputs:
        # x: concat (x, "zero")
        # y: concat (z, "zero", y)
        # Both are length ndim_tot. "zero" is small zero mean noise.
        x, y = (torch.cat((x, pad_x),  dim=1),
                torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz, y),
                          dim=1))
        

        optimizer.zero_grad()


        # output: model(x) = (z_hat, zero_hat, y_hat) ?
        # size ndim_tot
        output = model(x)

        # y_short: concat (z, y)
        y_short = torch.cat((y[:, :ndim_z], y[:, -ndim_y:]), dim=1)

        # Ly((zero_hat, y_hat), (zero, y))
        l = lambd_predict * loss_fit(output[:, ndim_z:], y[:, ndim_z:])

        # (z_hat, y_hat)
        output_block_grad = torch.cat((output[:, :ndim_z],
                                       output[:, -ndim_y:].data), dim=1)

        # Lz((z_hat, y_hat), (z, y))
        l += lambd_latent * loss_latent(output_block_grad, y_short)
        l_tot += l.data.item()

        # Backward Ly and Lz.
        l.backward()

        # Regenerate yz padding, apparently.
        pad_yz = zeros_noise_scale * torch.randn(batch_size, ndim_tot -
                                                 ndim_y - ndim_z, device=device)
        # Is y + noise, is ndim_y
        y = y_clean + y_noise_scale * torch.randn(batch_size, ndim_y, device=device)

        # Is z_hat + noise
        orig_z_perturbed = (output.data[:, :ndim_z] + y_noise_scale *
                            torch.randn(batch_size, ndim_z, device=device))
        # Is cat (z_hat + noise, "zero", y + noise)
        y_rev = torch.cat((orig_z_perturbed, pad_yz,
                           y), dim=1)
        # Is cat (noise, "zero", y + noise)
        y_rev_rand = torch.cat((torch.randn(batch_size, ndim_z, device=device), pad_yz,
                                y), dim=1)
        
        # Predict (x, "zero") from both of the above.
        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        # Lx(x_hat, x)
        # x_hat comes from sampling X dist with random Z, and original GT y.
        l_rev = (
            lambd_rev
            * loss_factor
            * loss_backward(output_rev_rand[:, :ndim_x],
                            x[:, :ndim_x])
        )

        # Ly((x_hat, "zero"), (x, "zero"))
        # x_hat generated from z and y with noise should regress to original x.
        l_rev += lambd_predict * loss_fit(output_rev, x)
        
        # Backward Lx and 2nd Ly.
        l_tot += l_rev.data.item()
        l_rev.backward()

        for p in model.parameters():
            p.grad.data.clamp_(-15.00, 15.00)

        optimizer.step()

    return l_tot / batch_idx
