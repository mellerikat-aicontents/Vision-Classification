# Welcome to IC !

⚡ image/ground truth tabular 형태의 데이터에 대해 분류할 수 있는 AI 컨텐츠입니다. ⚡

[![Generic badge](https://img.shields.io/badge/release-v1.0.0-green.svg?style=for-the-badge)](http://링크)
[![Generic badge](https://img.shields.io/badge/last_update-2023.10.16-002E5F?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Generic badge](https://img.shields.io/badge/python-3.10.12-purple.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Generic badge](https://img.shields.io/badge/dependencies-up_to_date-green.svg?style=for-the-badge&logo=python&logoColor=white)](requirement링크)
[![Generic badge](https://img.shields.io/badge/collab-blue.svg?style=for-the-badge)](http://collab.lge.com/main/display/AICONTENTS)
[![Generic badge](https://img.shields.io/badge/request_clm-green.svg?style=for-the-badge)](http://collab.lge.com/main/pages/viewpage.action?pageId=2157128981)


## 데이터 준비
1. .png, .jpg 형태의 동일한 shape(1024x1024, 3채널와 같이)을 가진 이미지 데이터와 정답 label 데이터를 준비합니다.
2. 이미지 경로와 정답 label로 이루어진 tabular 형태의 Ground truth 파일을 준비합니다.
3. 각 label 유형마다 최소 100장 이상 데이터가 있어야 안정적인 모델을 생성할 수 있습니다.
4. 현재 multi-class는 지원하지만 multi-label(한 장의 사진이 여러 유형을 갖는 경우)은 지원하지 않습니다.
5. Ground truth 파일은 하기와 같은 형태로 준비하시면 됩니다.

| label | image_path |
| ------ | ------ |
|label1| /nas001/users/.../sample1.png |
|label2| /nas001/users/.../sample2.png |
|label1| /nas001/users/.../sample3.png |
|...| ... |

데이터 명세서 상세한 내용은 [Documentation](http://collab.lge.com/main/pages/viewpage.action?pageId=2181826421)를 참고해주세요.

샘플 데이터 설명: [Fashion mnist dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
 


## 주요 기능 소개
- 현재 가볍고 빠른 모델인 mobilenet V1과 mobilenet V3 기반의 고해상도 모델을 사용할 수 있습니다.
- tensorflow v2에서 지원하는 tf.data 기반으로 구성되어 빠르고 효율적인 학습이 가능합니다.
- [RandAugment(2019)](https://arxiv.org/abs/1909.13719)를 사용함으로써 새로운 데이터에 대한 대응력이 좋은 모델을 생성할 수 있습니다.
- 현재 GPU지원 관련하여 개발 중 입니다.

상세한 설명은 [알고리즘 설명](http://collab.lge.com/main/pages/viewpage.action?pageId=2181826454)을 참고해주세요. 

## Quick Install Guide


```
git clone http://mod.lge.com/hub/dxadvtech/aicontents/ic.git 
cd ic 

conda create -n ic python=3.10
conda init bash
conda activate ic 

#jupyter 사용시 ipykernel 추가 필요
#pip install ipykernel
#python -m ipykernel install --user --name ic 

source install.sh

```
혹시 conda activate가 안되는 경우 bash를 입력하여 말머리가 `(base)`로 변했는지 확인 후 실행하시면 됩니다.

## Quick Run Guide
- 아래 코드 블럭을 실행하면 IC가 실행되고 이때 자동으로 `alo/config/experimental_plan.yaml`을 참조합니다. 
```
cd alo
python main.py 
```
- IC 구동을 위해서는 분석 데이터에 대한 정보 및 사용할 IC 기능이 기록된 yaml파일이 필요합니다.  
- IC default yaml파일인 `alo/config/experimental_plan.yaml`의 설정값을 변경하여 분석하고 싶은 데이터에 IC을 적용할 수 있습니다.
- 필수적으로 수정해야하는 ***arguments***는 아래와 같습니다. 
***
external_path:  
&emsp;- *load_train_data_path*: ***~/example/train_data_folder/***  # 학습 데이터가 들어있는 폴더 경로 입력(csv 입력 X)  
&emsp;- *load_inference_data_path*: ***~/example/inference_data_folder/***  # 추론 데이터가 들어있는 폴더 경로 입력(csv 입력 X)  
user_parameters:  
&emsp;- train_pipeline:  
&emsp;&emsp;- step: input  
&emsp;&emsp;&emsp;args:  
&emsp;&emsp;&emsp;- *input_path*: ***train_data_folder***  # 학습 데이터가 들어있는 폴더  
&emsp;&emsp;&emsp;&emsp;*y_column*: ***label***  # ground truth 데이터의 Y컬럼 명  
&emsp;&emsp;&emsp;&emsp;*path_column*: ***image_path***  # ground truth 데이터의 이미지 경로 컬럼 명    
&emsp;&emsp;&emsp;&emsp;*label_names*: ***[label1, label2, ...]***  # 분류하고자 하는 유형명(예, OK, NG)    
&emsp;&emsp;&emsp;&emsp;...  
&emsp;&emsp;- step: train   
&emsp;&emsp;&emsp;args:   
&emsp;&emsp;&emsp;- *input_shape*: ***[28,28,1]***     # 이미지 형태  
&emsp;&emsp;&emsp;- *num_classes*: ***10***     # 분류하고자 하는 유형 개수  
&emsp;&emsp;&emsp;&emsp;...   
&emsp;- inference_pipeline:  
&emsp;&emsp;- step: input  
&emsp;&emsp;&emsp;args:   
&emsp;&emsp;&emsp;- *input_path*: ***inference_data_folder***  # 추론 데이터가 들어있는 폴더  
&emsp;&emsp;&emsp;&emsp;*y_column*: ***label***  # 분석 데이터의 Y컬럼 명  
&emsp;&emsp;&emsp;&emsp;*path_column*: ***image_path***  # ground truth 데이터의 이미지 경로 컬럼 명  
&emsp;&emsp;&emsp;&emsp;...  
***
- IC의 다양한 기능을 사용하고 싶으신 경우 [User Guide (IC)](http://collab.lge.com/main/pages/viewpage.action?pageId=2205803957)를 참고하여 yaml파일을 수정하시면 됩니다. 
- 학습 결과 모델 파일 저장 경로: `alo/.train_artifacts/models/train/`
- 학습 결과 파일 저장 경로: `alo/.train_artifacts/output/train/`
- 추론 결과 파일 저장 경로: `alo/.inference_artifacts/output/inference/`



## Sample notebook(개발중)
Jupyter 환경에서 Workflow 단계마다 asset을 실행하고 setting을 바꿔 실험할 수 있습니다. [~~Sample notebook 링크~~](http://mod.lge.com/hub/dxadvtech/aicontents/ic/-/blob/main/IC_asset_run_template.ipynb)
현재 notebook 파일은 configuration 수정이 안되고 확인 및 실행만 됩니다. 빠른 시일 내에 업데이트 예정입니다.

## 관련 Collab
[AICONTENTS](http://collab.lge.com/main/display/AICONTENTS)

## 요청 및 문의
담당자: 서윤지(yoonji.suh@lge.com)

신규 AI Contents나 추가 기능 요청을 등록하시면 검토 후 반영합니다  [Request CLM](http://clm.lge.com/issue/projects/AICONTENTS/summary)


