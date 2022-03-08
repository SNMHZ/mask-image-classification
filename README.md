# 마스크 착용 상태 분류
<img src="https://user-images.githubusercontent.com/83912849/157253363-e8d94fee-80ce-4929-b19d-6618c6d7308a.png" width="1200" height="220">

**'마스크 착용 상태 분류'는 [부스트캠프](https://boostcamp.connect.or.kr/)에서 진행한 Image classification 프로젝트입니다.**

  
## 프로젝트 개요
<img src="https://user-images.githubusercontent.com/83912849/157253556-8bd2b548-e3c9-4530-b53b-f4d8bb088e38.jpg" width="850" height="440">
<a href="http://www.freepik.com">Designed by pikisuperstar / Freepik</a>    


#### “카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단하는 Task”

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

**따라서, 우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다.** 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

### 데이터 셋
- 모든 데이터셋은 아시아인 남녀로 구성되어 있고 나이는 20대부터 70대까지 다양하게 분포하고 있습니다.
- 입력 값
    - 전체 사람 수: 4,500 명 (train : 2700 | eval : 1800)
    - 이미지 크기: 384 x 512 Image
    - 한 사람 당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]
- 결과 값 : 0~17 classes  

|Class|Mask|Gender|Age|
|:--:|:--:|:--:|:--:|
|0|Wear|Male|<30|
|1|Wear|Male|>=30 and <60|
|2|Wear|Male|>=60|
|3|Wear|Female|<30|
|4|Wear|Female|>=30 and <60|
|5|Wear|Female|>=60|
|6|Incorrect|Male|<30|
|7|Incorrect|Male|>=30 and <60|
|8|Incorrect|Male|>=60|
|9|Incorrect|Female|<30|
|10|Incorrect|Female|>=30 and <60|
|11|Incorrect|Female|>=60|
|12|Not wear|Male|<30|
|13|Not wear|Male|>=30 and <60|
|14|Not wear|Male|>=60|
|15|Not wear|Female|<30|
|16|Not wear|Female|>=30 and <60|
|17|Not wear|Female|>=60|
        
### 활용 장비 및 개발환경
- OS : Ubuntu 20.04 LTS, Windows
- IDE : VS Code
- GPU : Tesla V100 *Boostcamp 로부터 제공받은 서버
- 주 사용 언어 : Python 3.8.5
- Frameworks : Pytorch 1.7.1, torchvision, albumentations, etc.
- Co-op tools : github, notion, slack

## 프로젝트 수행
![image](https://user-images.githubusercontent.com/83912849/157257440-4fb8652c-5ed1-41cc-8aac-4e572b6586e2.png)
- 프로젝트 목적의 달성과 동시에 팀원 개개인의 성장을 목표로, U stage에서 학습했던 다양한 방법론들과 추가적으로 습득한 지식을 실습한다.
- 팀원 개인이 모델링이 가능한 상황이므로 각자 실험을 반복하고, 시도한 방법과 결과를 공유하여 성능을 높이는 형태로 협업한다.

## 프로젝트 팀 구성 및 실험
#### Team: BEST CHOICE
|ID|Name|@Github|Experiment|
|:--:|:--:|:--|:--:|
|T3044|김원섭|[@whattSUPkim](https://github.com/whattSUPkim)|[link](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-15/tree/main/practice/t3044)|
|T3058|김진수|[@KimJinSuPKNU](https://github.com/KimJinSuPKNU)|[link](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-15/tree/main/practice/t3058)|
|T3080|민태원|[@mintaewon](https://github.com/mintaewon)|[link](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-15/tree/main/practice/t3080)|
|T3146|이상목|[@SNMHZ](https://github.com/SNMHZ)|[link](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-15/tree/main/practice/t3146)|
|T3204|조민재|[@binyf](https://github.com/binyf)|[link](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-15/tree/main/practice/t3204)|

## Wrap Up Report
👉 [Wrap-Up Report](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-15/blob/main/practice/Wrap-Up%20Report.PDF)
