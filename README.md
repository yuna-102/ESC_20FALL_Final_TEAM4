ESC_20FALL_Final_TEAM4
=======================
# CNN을 이용한 MR(Movie Review) 데이터 감성 분석

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
