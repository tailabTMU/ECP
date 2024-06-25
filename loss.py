################## loss func. #####################
from utils import *

def relu_evidence(y):
    return F.relu(y)

def exp_evidence(y):
    return torch.exp(torch.clamp(y/10, -10, 10))

def softplus_evidence(y):
    return F.softplus(y)

def pignistic(num_classes, output=None, prior='softmax', base_rate=None, device=None):
#     if not device:    # not needed!
#         device = get_device()
    if prior == 'softmax':
        if output is None:
            raise Exception("No output tensor is provided to generate softmax pignistic probabilities!!!")
        else:
            output = output.to(device)
            prior_prob = F.softmax(output, dim=1)
    elif prior == 'uniform':
        prior_prob = torch.ones((1, num_classes), dtype=torch.float32, device=device) / num_classes
    else:
        if base_rate is None:
            raise Exception("No base rate tensor is provided to generate pignistic probabilities!!!")
        else:
            prior_prob = base_rate
    pig = num_classes * prior_prob
    return prior_prob, pig

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

# def kl_divergence2(alpha, y, num_classes, device=None):
#     if not device:
#         device = get_device()
#     ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
#     sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
#     first_term = (
#         torch.lgamma(sum_alpha)
#         - torch.lgamma(alpha).sum(dim=1, keepdim=True)
#         + torch.lgamma(ones).sum(dim=1, keepdim=True)
#         - torch.lgamma(ones.sum(dim=1, keepdim=True))
#     )
#     second_term = (
#         (alpha - ones)
#         .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
#         .sum(dim=1, keepdim=True)
#     )
#     kl = first_term + second_term
#     return kl

def loglikelihood_loss(y, alpha, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood

# to be modified...
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, alpha, prior_p, epoch_num, num_classes, annealing_step, match, risk_mode=None, 
             risk_mat=None, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    prior_p = prior_p.to(device)    # softmax probabilities
    match = match.to(device) 
    pig_prob = num_classes * prior_p
    evid_mat = alpha - 1  

    S = torch.sum(alpha, dim=1, keepdim=True)
    
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    if risk_mode == 'objective':
        kappa = 1 / torch.max(risk_mat).item()    ## or annealing_coef
        y_arg = torch.argmax(y, dim=1, keepdim=True)
        risk_term = risk_mat[torch.flatten(y_arg).tolist()] * (evid_mat + pig_prob)
        risk = kappa * torch.sum(risk_term, dim=1, keepdim=True)
    elif risk_mode == 'subjective':
        baserate, _ = pignistic(num_classes, prior='uniform', device=device)
        p = alpha / S
        p = p.to(device)
        uncer = num_classes / S
        belief = evid_mat / S
        evid = (match * (1 / ((evid_mat + pig_prob) + 1e-8))) + ((1 - match) * torch.ones(alpha.size(0), num_classes, device=device)) 
        #######    old implementation of DC  ############
#         p_actual = torch.sum(p * y, dim=1, keepdim=True)
#         dc_term = ((1 - uncer) * (1 - p_actual))         # (match * (1 / (1e-8 + ((1 - uncer) * (1 - p_actual)))))
        #######    NEW implementation of DC  ############
        b_nonactual = torch.sum((1 - y) * belief, dim=1, keepdim=True)
        dc_term = ((1 - uncer)**2) * b_nonactual * (1 - match)    # (match * (1 / (1e-8 + ((1 - uncer) * (1 - p_actual)))))
        alpha_pp = (1 - match) * p       # match * p
        variance = ((alpha_pp * (1 - alpha_pp)) / (S + 1)) # (1 / (S + 1))  
        coeff = 1
        term1 = (coeff * (-1) * ((dc_term)**2) * torch.log(1 - dc_term + 1e-8))  # + (match * torch.ones(alpha.size(0), 1, device=device))
#         term2 = (1 - coeff) * torch.sum(((1 - variance)**2) * torch.log(variance + 1e-8), dim=1, keepdim=True) * (1 - match)
#         term2 = ((1 - coeff) * match * torch.sum((variance), dim=1, keepdim=True)) # + ((1 - match) * torch.ones(alpha.size(0), 1, device=device))
#         risk = torch.sum(((-1 * term1) + term2) * (evid_mat + pig_prob), dim=1, keepdim=True)
#         loss2 = (term1 + term2) * evid
#         loss2 = term1
#         loss2 = term1 + term2
        ############# risk calibration ######################  
        alpha_p = (1 - match) * p
        evid_m = (match * (1 / ((evid_mat + pig_prob) + 1e-8))) #+ ((1 - match) * torch.ones(alpha.size(0), num_classes, device=device))
        label_cost = (prior_p * (alpha_p**2)) / ((baserate * (-1) * torch.log(alpha_p + 1e-8)) + 1e-8)
        label_cost = label_cost.to(device)
        y_freq = torch.sum(y, dim=0, keepdim=True)
        rho = y_freq / alpha.size(0) 
        u_miss = ((1 - match) * (S / num_classes)) 
        subj_cost = ((u_miss * rho * label_cost)) #+ evid_m #+ (match * torch.ones(alpha.size(0), num_classes, device=device))) * evid_m
        subj_cost = (1 - y) * (subj_cost)  
        risk_term = subj_cost * (evid_mat + pig_prob)
        kappa = torch.min(torch.tensor(1.0, dtype=torch.float32), torch.tensor(epoch_num / 15, dtype=torch.float32))
        risk = (torch.sum(risk_term, dim=1, keepdim=True) * kappa) # + loss2       
        #####################################################
    else:
        risk = 0
    return A + kl_div + risk

# to be modified...
def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, evid_func=relu_evidence, device=None):
    if not device:
        device = get_device()
    evidence = evid_func(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss

# to be modified...
def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, evid_func=relu_evidence, device=None):
    if not device:
        device = get_device()
    evidence = evid_func(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss

def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, match, evid_func=relu_evidence, 
    prior='softmax', b_rate=None, risk_mode=None, r_mat=None, device=None):
    if not device:
        device = get_device()
    evidence = evid_func(output)
    alpha = evidence + 1
    prior_prob, _ = pignistic(num_classes, output, prior, b_rate, device)
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, prior_prob, epoch_num, num_classes, annealing_step, match, risk_mode, 
            r_mat, device)
    )
    return loss
    
