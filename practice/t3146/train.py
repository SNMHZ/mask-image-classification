import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader


def train_one_epoch(model: nn.Module, optimizer, data_loader: DataLoader, device, epoch):
    model.train()
    model.zero_grad()
    tqdm_dataloader = tqdm(data_loader)
    total_batch = 0
    for targets in tqdm_dataloader:
        images = targets['image'].to(device)
        labels = targets['label'].to(device)
        total_batch += len(images)

        logits = model(images)
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_dataloader.set_description(
            f"Epoch {epoch + 1}, lr is {optimizer.param_groups[0]['lr']:.6f} loss {loss.item():.3f}")

# Mission

# 1. 우리가 중간중간 모델의 결과를 저장하고 싶은 경우가 있습니다. 
# 학습되는 중간에 좋은 성능의 모델이 발견이 되었으면 모델을 저장하는 것이 좋은데요. 
# 이러한 체크포인트 저장 과정을 Evaluation에 추가해서 체크포인트 모델이 저장되도록 설계하세요.

# 2. 모델 평가에 Metric을 올바르게 사용하는 것도 중요합니다. 
# Evaluation 과정에서 모델 학습 결과를 Loss와 Accuracy만 보는것 보다는 다른 Metric도 참고하는 것도 좋습니다. 
# F1-score Metric 을 구현 or 코드를 찾아서 추가하고, 매 에폭마다 결과를 확인할 수 있도록 설계하세요.

# 3. Training Process에 어느정도 익숙해지셨다면 Gradient Accumulation을 내 프로젝트에 적용시켜보세요. 
# 적용한다고 반드시 성능이 올라가는 것은 아닐겁니다. 
# 적용을 했다면 전후 성능 비교를 해봐야겠지요?
# (코드는 강의자료에 있습니다.)

# 4. Optimizer의 스케줄러가 종류에 따라 어떻게 Learning rate를 변경해 가는지, 옵션에 따라 또 어떻게 스케쥴링하게 되는지 확인해보세요. 
# 대표적으로 많이 사용하는 SGD 와 Adam Optimizer를 각각 구현해 보고 비교해 봅시다. 
# 아래 Further Reading에 좋은 예제가 있습니다. 

# 5. 새로운 Loss를 만들어 봅시다. 
# 새로운 Loss를 적용해서 학습에 사용해보겠습니다. 
# (Label smoothing, Focal Loss, F1 Loss 등) 무엇이든지 상관없습니다. 
# 구현 후 학습에 활용해서 비교해보세요!