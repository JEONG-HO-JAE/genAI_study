import numpy as np
from losses.dsm import dsm_score_estimation
import logging
import torch
import os
import shutil
import tensorboardX
import torch.optim as optim
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from models.scorenet import ResScore

__all__ = ['ScoreNetRunner']

class ScoreNetRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999))
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))
        
    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)
    
    def train(self):
        transform = transforms.Compose([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor()
        ])

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                   transform=transform)
            
        elif self.config.data.dataset == 'MNIST':
            dataset = MNIST(os.path.join(self.args.run, 'datasets', 'mnist'), train=True, download=True,
                            transform=transform)
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()  # 현재 랜덤 상태 저장
            np.random.seed(2019)  # 랜덤 시드를 고정하여 동일한 데이터셋 분할이 되도록 설정
            np.random.shuffle(indices)  # 데이터셋 인덱스 셔플
            np.random.set_state(random_state)  # 이전 랜덤 상태 복구
            train_indices, test_indices = indices[:int(num_items * 0.8)], indices[int(num_items * 0.8):]  # 80% train, 20% test
            test_dataset = Subset(dataset, test_indices)  # test 데이터셋을 서브셋으로 분리
            dataset = Subset(dataset, train_indices)  # train 데이터셋을 서브셋으로 분리
        
        elif self.config.data.dataset == 'CELEBA':
            dataset = ImageFolder(root=os.path.join(self.args.run, 'datasets', 'celeba'),
                                transform=transforms.Compose([
                                    transforms.CenterCrop(140),  # 이미지를 중앙에서 140x140으로 자름
                                    transforms.Resize(self.config.data.image_size),  # 설정된 크기로 리사이즈
                                    transforms.ToTensor(),  # Tensor로 변환
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 데이터를 [-1, 1] 범위로 정규화
                                ]))
            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()  # 랜덤 상태 저장
            np.random.seed(2019)  # 시드 고정
            np.random.shuffle(indices)  # 셔플
            np.random.set_state(random_state)  # 이전 랜덤 상태 복구
            train_indices, test_indices = indices[:int(num_items * 0.7)], indices[int(num_items * 0.7):int(num_items * 0.8)]  # 70% train, 10% test
            test_dataset = Subset(dataset, test_indices)  # test 데이터셋 서브셋
            dataset = Subset(dataset, train_indices)  # train 데이터셋 서브셋

        # DataLoader를 사용하여 데이터셋을 배치 단위로 로드합니다.
        dataloader = DataLoader(dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.config.training.batch_size, shuffle=True, num_workers=4)

        # test_loader에서 batch iterator를 만듭니다.
        test_iter = iter(test_loader)

        # 입력 차원을 설정합니다. (이미지의 해상도 * 채널 수)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels

        # TensorBoard 경로 설정 및 초기화
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)  # 이전 TensorBoard 로그 파일 삭제
        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)  # TensorBoard 로깅 객체 생성

        # 스코어 네트워크(ResScore) 초기화 및 GPU에 할당
        score = ResScore(self.config).to(self.config.device)

        # 옵티마이저 설정 (Adam, RMSProp, SGD 중 하나)
        optimizer = self.get_optimizer(score.parameters())

        # 훈련을 이어서 진행할지 여부 확인
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))  # 체크포인트 불러오기
            score.load_state_dict(states[0])  # 모델 파라미터 복원
            optimizer.load_state_dict(states[1])  # 옵티마이저 상태 복원
            
        step = 0  # 현재 스텝 초기화

        # 훈련 시 사용할 노이즈 표준편차(sigma)
        sigma = self.config.training.noise_std
        
        for epoch in range(self.config.training.n_epochs):
            for i, (X, y) in enumerate(dataloader):
                step += 1  # 스텝 증가

                X = X.to(self.config.device)
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)  # 데이터에 logit 변환 적용

                scaled_score = lambda x: score(x)

                if self.config.training.algo == 'dsm':
                    loss = dsm_score_estimation(scaled_score, X, sigma=self.config.training.noise_std)
                    
                optimizer.zero_grad()
                loss.backward()  
                optimizer.step() 

                tb_logger.add_scalar('loss', loss, global_step=step)
                tb_logger.add_scalar('sigma', sigma, global_step=step)
                logging.info("step: {}, loss: {}, sigma: {}".format(step, loss.item(), sigma))

                # 최대 스텝 도달 시 훈련 종료
                if step >= self.config.training.n_iters:
                    return 0

                # 매 100 스텝마다 테스트 데이터셋에 대한 손실 계산
                if step % 100 == 0:
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)  # 테스트 셋이 끝났으면 다시 반복자 초기화
                        test_X, test_y = next(test_iter)

                    test_X = test_X.to(self.config.device)  # 테스트 데이터도 GPU로 이동
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)  # 테스트 데이터에도 logit 변환 적용

                    # 테스트 셋에 대해 알고리즘 선택 및 손실 계산
                    if self.config.training.algo == 'ssm':
                        test_X += torch.randn_like(test_X) * self.config.training.noise_std
                        test_loss, *_ = sliced_score_estimation_vr(scaled_score, test_X.detach(), n_particles=1)
                    elif self.config.training.algo == 'dsm':
                        test_loss = dsm_score_estimation(scaled_score, test_X, sigma=self.config.training.noise_std)

                    # TensorBoard에 테스트 셋 손실 기록
                    tb_logger.add_scalar('test_loss', test_loss, global_step=step)
                    
                # 스냅샷 주기에 따라 체크포인트 저장
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),  # 모델 상태 저장
                        optimizer.state_dict()  # 옵티마이저 상태 저장
                    ]
                    # 체크포인트를 지정된 경로에 저장
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

                # 최대 스텝 도달 시 종료
                if step == self.config.training.n_iters:
                    return 0    