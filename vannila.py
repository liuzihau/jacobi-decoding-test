def speculative_decoding(
    net: Module,
    small_net: Module,
    prompt: Tensor,
    seq_len: int,
    gamma: int = 5,
    temperature = 1.,
    filter_thres = 0.9,
    lenience = 1.,
    pad_id = 0
):
    """
    eq. algorithm 1 in paper https://arxiv.org/abs/2211.17192
    """

    batch, prompt_seq_len, out, device = *prompt.shape, prompt.clone(), prompt.device
    sample_num_times = max(0, seq_len - prompt_seq_len)

    cache = None
    small_cache = None

    num_steps = 0
    total_accepted = 0

    batch_range = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    seq_lens = torch.full((batch,), prompt_seq_len, device = device, dtype = torch.long)

    while (seq_lens < seq_len).any():

        # predict with smaller network

        all_small_logits = []
        q_sampled_out = []

        for _ in range(gamma):
            small_logits, small_cache = small_net(
                out,
                seq_start_pos = out.shape[-1] - seq_lens,
                cache = small_cache,
                return_cache = True
            )

            small_logits = small_logits[:, -1]

            small_logits = top_k(small_logits, thres = filter_thres)
            all_small_logits.append(small_logits)

            sample = gumbel_sample(small_logits, temperature = temperature, dim = -1)
            out = torch.cat((out, sample[..., None]), dim = -1)
            seq_lens += 1

            q_sampled_out.append(rearrange(sample, 'b -> b 1 1'))

        q_sampled_out = torch.cat(q_sampled_out, dim = -2)
        small_logits = torch.stack(all_small_logits, dim = -2)

        # verify with larger network

        logits, cache = net(
            out,
            seq_start_pos = out.shape[-1] - seq_lens,
            cache = cache,
            return_cache = True
        )

        logits = logits[..., -(gamma + 1):, :]
        logits = top_k(logits, thres = filter_thres)

        # prob and prob of small model (p(x) and q(x) in algorithm 1)

        prob = safe_div(logits, temperature).softmax(dim = -1)
        small_prob = safe_div(small_logits, temperature).softmax(dim = -1)

        p, prob_next = prob[:, :-1], prob[:, -1]

        p = p.gather(-1, q_sampled_out)
        q = small_prob.gather(-1, q_sampled_out) * lenience

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]

        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        accepted = find_first_true_index(r > (p / q))

        total_accepted += accepted.float().mean()
        num_steps += 1

        num_rejected = gamma - accepted
        has_rejected = num_rejected > 0

        accepted = rearrange(accepted, 'b -> b 1')
        accepted.clamp_(max = gamma - 1)
        adjusted_prob = F.relu(prob[batch_range, accepted] - small_prob[batch_range, accepted])
        adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)
        adjusted_prob = rearrange(adjusted_prob, 'b 1 d -> b d')

        prob_next = torch.where(
            rearrange(has_rejected, '... -> ... 1'),
            adjusted_prob,
            prob_next
        )

        # do a bunch of slicing and align everything to the right, including kv caches

        seq_lens -= num_rejected
        max_seq_len = seq_lens.amax()
        curr_len = out.shape[-1]

        seq_arange = torch.arange(max_seq_len, device = device, dtype = torch.long) + (curr_len - max_seq_len)
        seq_offset_indices = seq_arange - num_rejected[..., None]

        cached_kv, _ = cache
        small_cached_kv, _ = small_cache

        if batch > 1:
            small_cached_kv = F.pad(small_cached_kv, (0, 0, 0, 1))  # account for small model being a token behind

            out = out[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b ... n d -> b n ... d')
            small_cached_kv = rearrange(small_cached_kv, 'b ... n d -> b n ... d')

            cached_kv = cached_kv[batch_range, seq_offset_indices]
            small_cached_kv = small_cached_kv[batch_range, seq_offset_indices]

            cached_kv = rearrange(cached_kv, 'b n ... d -> b ... n d')
            small_cached_kv = rearrange(small_cached_kv, 'b n ... d -> b ... n d')

            small_cached_kv = small_cached_kv[..., :-1, :]
        else:
            # if batch size of 1, just slice to max_seq_len
            out = out[..., :max_seq_len]
            cached_kv = cached_kv[..., :max_seq_len, :]
            small_cached_kv = small_cached_kv[..., :max_seq_len, :]

        cache = (cached_kv, None)
        small_cache = (small_cached_kv, None)

        # sample the additional token, one of the tricks in the paper to better bound the worst case

        next_token = torch.multinomial(prob_next, 1)

        out = torch.cat((out, next_token), dim = -1)
        seq_lens += 1

    # now left align

    num_pad_left = out.shape[-1] - seq_lens
    max_pad_left = num_pad_left.amax()
    out = F.pad(out, (0, max_pad_left), value = pad_id)

    seq_len_range = torch.arange(seq_len, device = device, dtype = torch.long)
    out = out[batch_range, seq_len_range + num_pad_left[..., None]]

    return out[..., prompt_seq_len:], total_accepted / num_steps

