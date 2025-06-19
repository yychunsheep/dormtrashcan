"# dormtrashcan" 
"# dormtrashcan" 
import cv2
import sys

# 載入訓練好的模型
cascade = cv2.CascadeClassifier("classifier/cascade.xml")

# 載入測試圖片
image_path = sys.argv[1] if len(sys.argv) > 1 else "images/test.jpg"
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 偵測垃圾桶
trashcans = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# 繪製結果
for (x, y, w, h) in trashcans:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Trashcan Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
