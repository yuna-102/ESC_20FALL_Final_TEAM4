ESC_20FALL_Final_TEAM4
=======================
# CNN을 이용한 MR(Movie Review) 데이터 감성 분석

## Requirements

~~~
  nltk==3.2.5.
  torch==1.7.0
~~~

## 1. Dataset
> Naver sentiment movie corpus v1.0
> |text|label|
> |------|----|
> |simplistic , silly and tedious .	|negative|
> |it's so laddish and juvenile , only teenage boys could possibly find it funny . |negative|
> |the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .|positive|
> |a masterpiece four years in the making .|positive|
> |together , tok and o orchestrate a buoyant , darkly funny dance of death . in the process , they demonstrate that there's still a lot of life in hong kong cinema . |positive|
> |interesting , but not compelling . |negative|
> |the action clich�s just pile up .|negative|
> |[a] rare , beautiful film .|positive|


## 2. Preprocessing
> 한글, 영문, 숫자, 괄호, 쉼표, 느낌표, 물음표, 작음따옴표, 역따옴표 제외한 나머지 모두 찾아서 공백(" ")으로 바꾸기
> |Preprocessing 전|Preprocessing 후|
> |------|----|
> |[a] rare , beautiful film . | a rare , beautiful film . |


## 3. Implementation
> python main.py --help
  parser.add_argument('--batch-size', default=50, type=int)
  parser.add_argument('--dropout', default=0.5, type=float)
  parser.add_argument('--epoch', default=20, type=int)
  parser.add_argument('--learning-rate', default=0.1, type=float)
  parser.add_argument("--mode", default="non-static", help="available models: rand, static, non-static")
  parser.add_argument('--num-feature-maps', default=100, type=int) 
  parser.add_argument("--pretrained-word-vectors", default="fasttext", help="available models: fasttext, Word2Vec")
  parser.add_argument("--save-word-vectors", action='store_true', default=False, help='save trained word vectors')
  parser.add_argument("--predict", action='store_true', default=False, help='classify your sentence')
  args = parser.parse_args()
  
    usage: main.py [-h] [--batch-size BATCH_SIZE] [--dropout DROPOUT] 
                    [--epoch EPOCH] [--learning-rate LEARNING_RATE]
                    [----predict PREDICT] [--mode MODE]
                    [--num-feature-maps NUM_FEATURE_MAPS]
                    [--pretrained-word-vectors PRETRAINED_WORD_VECTORS]
                    [--save-word-vectors SAVE_WORD_VECTORS]

    optional arguments:
      -h, --help            show this help message and exit
      --batch-size BATCH_SIZE
      --dropout DROPOUT
      --epoch EPOCH
      --learning-rate LEARNING_RATE
      --mode MODE           available models: rand, static, non-static,
      --num-feature-maps NUM_FEATURE_MAPS
      --pretrained-word-vectors           available models: fasttext, Word2Vec
      --save-word-vectors SAVE_WORD_VECTORS           default:False


## 4. Results
Baseline from the paper

> | Model | MR | 
> | ----- | -- | 
> | random | 76.1 | 
> | static | 81.0 | 
> | non-static | 81.5 | 


Re-implementation with Word2Vec and fasttext

> | Model | MR (Word2Vec) | MR (fasttext) |
> | ----- | -- | -- | 
> | random | *73.11 | *73.11 | 
> | static | 81.30 | 82.56 | 
> | non-static | 81.75| 82.65 |
* *no pre-trained word vector





