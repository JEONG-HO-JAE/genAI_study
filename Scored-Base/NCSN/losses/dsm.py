import torch

def dsm_score_estimation(scorenet, samples, sigma=0.01):
    perturbed_samples = samples + torch.randn_like(samples) * sigma  # \tilde{x} = x + 노이즈
    target = - 1 / (sigma ** 2) * (perturbed_samples - samples)  # 실제 스코어 = - (\tilde{x} - x) / \sigma^2
    scores = scorenet(perturbed_samples)  # 스코어 네트워크의 예측값 s_\theta (\tilde{x}, \sigma)
    target = target.view(target.shape[0], -1)  # 타겟 스코어 reshape
    scores = scores.view(scores.shape[0], -1)  # 예측된 스코어 reshape
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)  # L2 손실 계산

    return loss

def anneal_dsm_score_estimation(scorenet, samples, labels, sigmas, anneal_power=2.):
    # 각 샘플에 대응하는 시그마 값을 사용
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    
    # 샘플에 노이즈 추가 (x̃ = x + 노이즈)
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    
    # 타겟 스코어 계산 (스코어 = -(x̃ - x) / σ^2)
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    
    # 스코어 네트워크가 예측한 스코어 값
    scores = scorenet(perturbed_samples, labels)
    
    # 스코어와 타겟 스코어를 비교할 수 있도록 각각 reshape
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    
    # L2 손실을 사용한 스코어 손실 계산 후, 노이즈 단계에 따라 가중치를 줌
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    # 손실 값 평균 계산
    return loss.mean(dim=0)