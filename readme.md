###数据来源

**原始数据信息：**风险因素监测系统(BRFSS)是由CDC每年收集的与健康相关的电话调查。每年，该调查收集来自超过40万名美国人对与健康相关的风险行为、慢性健康状况以及预防服务使用的回应。该调查自1984年以来每年都进行。对于这个项目，使用了Kaggle上可用于2015年的数据集的CSV文件。这个原始数据集包含了441,455个人的回应，共有330个特征。这些特征要么是直接向参与者提出的问题，要么是基于个人参与者回答的计算变量。
**data_process.py**即是对原始数据文件的处理脚本
**原始数据集网址**：https://www.kaggle.com/datasets/cdc/behavioral-risk-factor-surveillance-system?select=2015.csv

**处理后的数据集网址**：https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
这部分数据集就是我们所用到的：**diabetes_binary_5050split_health_indicators_BRFSS2015.csv**

###脚本的执行
data文件中的**data_view.ipynb**文件为一些数据可视化处理，可以依次执行查看结果

**start_ml.sh，start_dl_mlp.sh，start_dl_cnn.sh**在linux终端执行后可以分别在machine_learning和deep_learning中得到SVM，KNN，决策树，随机森林，以及MLP和CNN的对数据集的预测指标结果，并保存为txt文件

执行过程：
bash start_ml.sh
bash start_dl_mlp.sh
bash start_dl_cnn.sh

**feature_select.ipynb**用于特征选择，内含不同的特征选择方法，并在data里生成对应的新的csv数据集文件，包括chi，pearson和f_classifi方式

**用到的python库以及脚本**
CUDA版本：11.4  显卡版本：1080Ti 250W 12G
Package                       Version
----------------------------- -------------------
absl-py                       1.0.0
aiosignal                     1.2.0
argon2-cffi                   21.3.0
argon2-cffi-bindings          21.2.0
astunparse                    1.6.3
attrs                         21.4.0
backcall                      0.2.0
backports.functools-lru-cache 1.6.4
beautifulsoup4                4.10.0
bleach                        4.1.0
boto3                         1.21.22
botocore                      1.24.22
Bottleneck                    1.3.4
brotlipy                      0.7.0
cached-property               1.5.2
cachetools                    5.0.0
certifi                       2021.10.8
cffi                          1.15.0
charset-normalizer            2.0.4
click                         8.0.4
cryptography                  36.0.0
cycler                        0.11.0
debugpy                       1.5.1
decorator                     5.1.1
defusedxml                    0.7.1
distlib                       0.3.4
entrypoints                   0.4
filelock                      3.6.0
flatbuffers                   2.0
fonttools                     4.31.1
frozenlist                    1.3.0
gast                          0.5.3
google-auth                   2.6.5
google-auth-oauthlib          0.4.6
google-pasta                  0.2.0
grpcio                        1.43.0
h5py                          3.6.0
htmlmin                       0.1.12
huggingface-hub               0.5.1
idna                          3.3
ImageHash                     4.2.1
imbalanced-learn              0.10.1
imblearn                      0.0
importlib-metadata            4.11.3
importlib-resources           5.4.0
ipykernel                     6.9.2
ipython                       7.32.0
ipython-genutils              0.2.0
ipywidgets                    7.7.0
jedi                          0.18.1
Jinja2                        3.0.3
jmespath                      1.0.0
joblib                        1.2.0
jsonschema                    4.4.0
jupyter-client                7.1.2
jupyter-core                  4.9.2
jupyterlab-pygments           0.1.2
jupyterlab-widgets            1.1.0
keras                         2.8.0
Keras-Preprocessing           1.1.2
kiwisolver                    1.4.0
libclang                      13.0.0
Markdown                      3.3.6
MarkupSafe                    2.0.1
matplotlib                    3.5.1
matplotlib-inline             0.1.3
missingno                     0.5.1
mistune                       0.8.4
mkl-fft                       1.3.1
mkl-random                    1.2.2
mkl-service                   2.4.0
msgpack                       1.0.3
multimethod                   1.7
nbclient                      0.5.13
nbconvert                     6.4.4
nbformat                      5.2.0
nest-asyncio                  1.5.4
networkx                      2.6.3
nltk                          3.7
notebook                      6.4.10
numexpr                       2.8.1
numpy                         1.21.2
oauthlib                      3.2.0
opt-einsum                    3.3.0
packaging                     21.3
pandas                        1.3.5
pandas-profiling              3.1.0
pandocfilters                 1.5.0
parso                         0.8.3
pexpect                       4.8.0
phik                          0.12.1
pickleshare                   0.7.5
Pillow                        9.0.1
pip                           21.2.2
platformdirs                  2.5.2
prometheus-client             0.13.1
prompt-toolkit                3.0.27
protobuf                      3.20.0
psutil                        5.9.0
ptyprocess                    0.7.0
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycparser                     2.21
pydantic                      1.9.0
Pygments                      2.11.2
pyOpenSSL                     22.0.0
pyparsing                     3.0.7
pyrsistent                    0.18.1
PySnooper                     1.1.1
PySocks                       1.7.1
python-dateutil               2.8.2
python-louvain                0.15
pytorch-pretrained-bert       0.6.2
pytorch-transformers          1.2.0
pytz                          2021.3
PyWavelets                    1.3.0
PyYAML                        6.0
pyzmq                         22.3.0
ray                           1.12.0
regex                         2022.3.15
requests                      2.27.1
requests-oauthlib             1.3.1
rsa                           4.8
s3transfer                    0.5.2
sacremoses                    0.0.49
scikit-learn                  1.0.2
scipy                         1.7.3
seaborn                       0.11.2
Send2Trash                    1.8.0
sentencepiece                 0.1.96
setuptools                    58.0.4
six                           1.16.0
sklearn                       0.0
soupsieve                     2.3.1
tabulate                      0.8.9
tangled-up-in-unicode         0.1.0
tensorboard                   2.8.0
tensorboard-data-server       0.6.1
tensorboard-plugin-wit        1.8.1
tensorboardX                  2.5
tensorflow                    2.8.0
tensorflow-io-gcs-filesystem  0.24.0
termcolor                     1.1.0
terminado                     0.13.3
testpath                      0.6.0
tf-estimator-nightly          2.8.0.dev2021122109
threadpoolctl                 3.1.0
tokenizers                    0.12.1
torch                         1.11.0
torch-cluster                 1.6.0
torch-geometric               2.0.4
torch-scatter                 2.0.9
torch-sparse                  0.6.13
torch-spline-conv             1.2.1
torchaudio                    0.11.0
torchdata                     0.3.0
TorchSnooper                  0.8
torchtext                     0.12.0
torchvision                   0.12.0
tornado                       6.1
tqdm                          4.64.0
traitlets                     5.1.1
transformers                  4.18.0
typing-extensions             3.10.0.2
urllib3                       1.26.8
virtualenv                    20.14.1
visions                       0.7.4
wcwidth                       0.2.5
webencodings                  0.5.1
Werkzeug                      2.1.1
wheel                         0.37.1
widgetsnbextension            3.6.0
wordcloud                     1.8.1
wrapt                         1.14.0
yacs                          0.1.6
zipp                          3.7.0




# BRFSS2015-prediction-of-diabetes
