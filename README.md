依赖库

```
keras
numpy
scipy
opencv-python
scikit-image
pillow
tensorflow
```



文件夹 `hyperlpr` 需要放到 Python 虚拟环境路径的 `Lib` 文件夹下



代码示例：

```python
from hyperlpr import pipline as  pp
import cv2
image = cv2.imread("./images_rec/2.jpg")
image,res  = pp.SimpleRecognizePlate(image)
print(res)
```





关键函数：

```python
def SimpleRecognizePlate(image):
    t0 = time.time()
    images = detect.detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)
    res_set = []
    for j,plate in enumerate(images):
        plate, rect, origin_plate  =plate
        # plate = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
        plate  =cv2.resize(plate,(136,36*2))
        t1 = time.time()

        ptype = td.SimplePredict(plate)
        if ptype>0 and ptype<5:
            plate = cv2.bitwise_not(plate)

        image_rgb = fm.findContoursAndDrawBoundingBox(plate)

        image_rgb = fv.finemappingVertical(image_rgb)
        cache.verticalMappingToFolder(image_rgb)
        print("e2e:", e2e.recognizeOne(image_rgb))
        image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)

        # image_gray = horizontalSegmentation(image_gray)
        # cv2.imshow("image_gray",image_gray)
        # cv2.waitKey()

        cv2.imwrite("./"+str(j)+".jpg",image_gray)
        # cv2.imshow("image",image_gray)
        # cv2.waitKey(0)
        print("校正",time.time() - t1,"s")
        # cv2.imshow("image,",image_gray)
        # cv2.waitKey(0)
        t2 = time.time()
        val = segmentation.slidingWindowsEval(image_gray)
        # print val
        print("分割和识别",time.time() - t2,"s")
        if len(val)==3:
            blocks, res, confidence = val
            if confidence/7>0.7:
                image =  drawRectBox(image,rect,res)
                res_set.append(res)
                for i,block in enumerate(blocks):

                    block_ = cv2.resize(block,(25,25))
                    block_ = cv2.cvtColor(block_,cv2.COLOR_GRAY2BGR)
                    image[j * 25:(j * 25) + 25, i * 25:(i * 25) + 25] = block_
                    if image[j*25:(j*25)+25,i*25:(i*25)+25].shape == block_.shape:
                        pass


            if confidence>0:
                print("车牌:",res,"置信度:",confidence/7)
            else:
                pass

                # print "不确定的车牌:", res, "置信度:", confidence

    print(time.time() - t0,"s")
    return image,res_set
```

