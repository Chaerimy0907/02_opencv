import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('../img/dog.jpg')

# 이미지 사이즈 조절
img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)

# 그레이 스케일로 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 블러로 노이즈 제거
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 오츠 알고리즘으로 자동 이진화
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 마스크를 보정
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# 마스크 반전(흰색=객체, 검정=배경)
mask_inv = cv2.bitwise_not(mask)

# 객체만 추출
obj = cv2.bitwise_and(img, img, mask=mask_inv)

# 배경을 흑백으로 변환
background = cv2.bitwise_and(img, img, mask=mask)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

# 객체 + 흑백 배경 합성
result = cv2.add(obj, background)

#print(img.shape)
cv2.imshow('Origianl', img)
cv2.imshow('Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()