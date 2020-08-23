import os
import shutil

import cv2


def generator(data):
    '''
    打开摄像头，读取帧，检测该帧图像中的人脸，并进行剪切、缩放
    生成图片满足以下格式：
    1.灰度图，后缀为 .png
    2.图像大小相同
    params:
        data:指定生成的人脸数据的保存路径
    '''

    name = input('my name:')
    # 如果路径存在则删除路径
    path = os.path.join(data, name)
    if os.path.isdir(path):
        shutil.rmtree(path)
    # 创建文件夹
    os.mkdir(path)
    # 创建一个级联分类器
    face_casecade = cv2.CascadeClassifier('g:/haarcascade_frontalface_default.xml')
    # 打开摄像头
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Dynamic')
    # 计数
    count = 1

    while (True):
        # 读取一帧图像
        ret, frame = camera.read()
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸检测
            face = face_casecade.detectMultiScale(gray_img, 1.3, 5)
            for (x, y, w, h) in face:
                # 在原图上绘制矩形
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 调整图像大小
                new_frame = cv2.resize(frame[y:y + h, x:x + w], (92, 112))
                # 保存人脸
                cv2.imwrite('%s/%s.png' % (path, str(count)), new_frame)
                count += 1
            cv2.imshow('Dynamic', frame)
            # 按下q键退出
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()
generator('./')